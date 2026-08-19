"""通配符 + 结构化 shell 规则匹配器。

职责：
- 在 Rule 数据结构的 action 通配符匹配基础上，
  对 shell 命令额外支持 command / subcommand / flags 级结构化匹配
- 非 shell 工具退化为 resource_pattern 通配符匹配

权限引擎与 grant 匹配复用此模块；它不持有运行时上下文。

RuleMatcher 是无状态纯函数集合，不持有任何上下文。
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from collections.abc import Sequence

from .permission_model import Action, Rule


def _compile_wildcard(pattern: str, *, cross_path: bool = False) -> re.Pattern:
    """将 glob 风格通配符编译为正则。

    支持 *（任意字符）、?（单字符）。
    cross_path=True 时 * 匹配任意字符包括 /。
    """
    escaped = re.escape(pattern)
    if cross_path:
        escaped = escaped.replace(r"\*", ".*")  # * → 任意字符
        if escaped.endswith(r"\ .*"):
            escaped = escaped[: -len(r"\ .*")] + r"(?:\ .*)?"
    else:
        escaped = escaped.replace(r"\*\*", ".*")  # ** → 跨越路径
        escaped = escaped.replace(r"\*", "[^/]*")  # * → 非分隔符
    escaped = escaped.replace(r"\?", ".")  # ? → 单字符
    return re.compile(f"^{escaped}$")


def _wildcard_match(value: str, pattern: str, *, cross_path: bool = False) -> bool:
    """通配符匹配。

    cross_path=True 时 * 匹配任意字符（用于 shell 命令文本匹配）。
    """
    value = value.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    if cross_path:
        return bool(_compile_wildcard(pattern, cross_path=True).match(value))
    if "/" in pattern or "?" in pattern:
        return bool(_compile_wildcard(pattern).match(value))
    return fnmatch.fnmatch(value, pattern)


def _shlex_extract(command: str) -> tuple[str | None, str | None, set[str]]:
    """用 shlex 从命令字符串中提取结构化信息。

    返回 (primary_command, subcommand, flags)。
    短 flag 的字符会排序归一化（-rf → -fr），以便与 flags_any 匹配。
    这是 Shell 分类不可用时的轻量降级路径。
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return (None, None, set())

    if not tokens:
        return (None, None, set())

    primary = tokens[0].lower()
    flags: set[str] = set()
    subcommand: str | None = None

    for token in tokens[1:]:
        if token.startswith("--"):
            # --flag=value 或 --flag
            flag = token.split("=", 1)[0].lower()
            flags.add(flag)
        elif token.startswith("-") and not token.startswith("--"):
            # -rf → 字符排序归一化为 -fr
            chars = sorted(token[1:].lower())
            flags.add("-" + "".join(chars))
        elif subcommand is None and not token.startswith("-"):
            # 第一个非 flag token 是子命令
            subcommand = token.lower()

    return (primary, subcommand, flags)


def _action_wildcard_matches(tool_name: str, action_pattern: str) -> bool:
    """
    工具名通配符匹配
    """
    if action_pattern == "*":
        return True
    return _wildcard_match(tool_name, action_pattern)


def matches(
    action: Action,
    rule: Rule,
    *,
    shell_command: str | None = None,
    primary_command: str | None = None,
    subcommand: str | None = None,
    flags: set[str] | None = None,
) -> bool:
    """判断 action 是否匹配 rule。

    支持两种调用方式：
    1. 已提取的结构化 shell 字段传入
    2. 传入 shell_command 原始字符串，内部用 shlex 提取（降级路径）

    对于非 shell 工具（capability != "shell"），只匹配 action + resource_pattern。
    """
    # 1. 工具名通配匹配
    if not _action_wildcard_matches(action.tool, rule.action):
        return False

    # 2. 非 shell → resource_pattern 通配匹配
    if action.capability != "shell":
        if rule.resource_pattern is not None:
            return any(
                _wildcard_match(t.value, rule.resource_pattern) for t in action.targets
            )
        # action 匹配且 rule 无额外路径约束 → 匹配
        return True

    # 3. shell → 结构化匹配
    # 提取结构化字段（优先用传入的，降级到 shlex）
    cmd: str | None = primary_command
    sub: str | None = subcommand
    flg: set[str] = set(flags or ())

    if shell_command is not None and (cmd is None or sub is None or not flg):
        extracted_cmd, extracted_sub, extracted_flg = _shlex_extract(shell_command)
        if cmd is None:
            cmd = extracted_cmd
        if sub is None:
            sub = extracted_sub
        if not flg:
            flg = extracted_flg

    # 3a. command 匹配
    if rule.command is not None:
        if cmd is None or not _wildcard_match(cmd, rule.command):
            return False

    # 3b. subcommand 匹配（精确）
    if rule.subcommand is not None:
        if sub is None or rule.subcommand != sub:
            return False

    # 3c. subcommand_in 匹配（集合）
    if rule.subcommand_in is not None:
        if sub is None or sub not in rule.subcommand_in:
            return False

    # 3d. flags_any 匹配（含任一即可）
    if rule.flags_any is not None:
        if not (flg & rule.flags_any):
            return False

    # 3e. flags_all 匹配（含全部才可）
    if rule.flags_all is not None:
        if not (rule.flags_all <= flg):
            return False

    # 3f. resource_pattern 额外约束
    if rule.resource_pattern is not None:
        return any(
            _wildcard_match(t.value, rule.resource_pattern, cross_path=True)
            for t in action.targets
        )

    return True


def first_match(
    action: Action,
    rules: Sequence[Rule],
    *,
    shell_command: str | None = None,
    primary_command: str | None = None,
    subcommand: str | None = None,
    flags: set[str] | None = None,
) -> Rule | None:
    """findLast 语义：返回最后一条匹配的规则。

    如果规则列表为空或全部不匹配，返回 None。
    """
    matched: Rule | None = None
    for rule in rules:
        if matches(
            action,
            rule,
            shell_command=shell_command,
            primary_command=primary_command,
            subcommand=subcommand,
            flags=flags,
        ):
            matched = rule
    return matched


def evaluate(
    action: Action,
    rules: Sequence[Rule],
    *,
    fallback: str = "ask",
    shell_command: str | None = None,
    primary_command: str | None = None,
    subcommand: str | None = None,
    flags: set[str] | None = None,
) -> str:
    """在 ruleset 中评估 action，返回 allow / ask / deny。

    规则按顺序遍历，findLast 匹配。全部不匹配则返回 fallback。
    fallback 按模式分别为 plan="deny"、build="ask"、act="ask"。
    """
    matched = first_match(
        action,
        rules,
        shell_command=shell_command,
        primary_command=primary_command,
        subcommand=subcommand,
        flags=flags,
    )
    if matched is not None:
        return matched.effect
    return fallback


def merge_rulesets(
    user_rules: Sequence[Rule] | None,
    default_rules: Sequence[Rule],
) -> list[Rule]:
    """合并用户配置规则和默认规则。

    用户配置规则优先级高，放在后面（findLast 优先匹配用户规则）。
    用户未配置时使用全部默认规则。
    """
    if not user_rules:
        return list(default_rules)
    return list(default_rules) + list(user_rules)
