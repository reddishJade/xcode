"""Plan / Build / Act 的工具可见性策略与三态 ruleset 初始化。

PlanPolicy.filter_tools() 暴露只读工具，并允许文件工具维护计划文件。
初始化 MODE_DEFAULT_RULES，build 默认规则为透明的工具级 profile。
"""

from __future__ import annotations

from typing import Protocol

from xcode.ai.events import ToolCall

from ..config import ExecutionMode
from ..observability.permission_model import MODE_DEFAULT_RULES, Rule
from ..observability.permissions import PermissionDecision
from ..skills import ToolSpec


class ExecutionPolicy(Protocol):
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]: ...

    def check_call(self, call: ToolCall) -> PermissionDecision: ...


class ExecutionModeState:
    """管理当前执行模式和 plan 模式超时状态。"""

    def __init__(self, max_plan_turns: int = 8) -> None:
        self._current_mode: ExecutionMode = "act"
        self._plan_enter_step = 0
        self._max_plan_turns = max_plan_turns

    @property
    def current_mode(self) -> ExecutionMode:
        return self._current_mode

    def set_mode(self, mode: ExecutionMode) -> None:
        """设置当前执行模式。"""
        self._current_mode = mode
        if mode == "plan":
            self._plan_enter_step = 0

    def check_plan_timeout(self) -> bool:
        """检查 plan 模式是否超时，超时则自动切换到 build。"""
        if self._current_mode != "plan":
            return False
        self._plan_enter_step += 1
        if self._plan_enter_step < self._max_plan_turns:
            return False
        self._plan_enter_step = 0
        self._current_mode = "build"
        return True

    def filter_tools(self, registry: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        """根据当前模式过滤工具集。"""
        return registry_for_mode(registry, self._current_mode)


class PlanPolicy:
    """plan: 只读分析，可维护 .xcode/plans/*.md 计划文件。"""

    _READ_ONLY_TOOLS = frozenset({
        "read_file", "glob_files", "find_files", "list_dir", "grep_search",
        "search_tools", "webfetch", "websearch", "question", "search_memory",
    })

    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tuple(
            tool
            for tool in tools
            if tool.name in self._READ_ONLY_TOOLS or tool.name in {"write_file", "edit_file"}
        )

    def check_call(self, call: ToolCall) -> PermissionDecision:
        # plan 模式的实际写入边界由 RuleMatcher + fallback=deny 执行。
        return "allow"


class BuildPolicy:
    """build: 读写工具和 shell 默认放行。"""

    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tools

    def check_call(self, call: ToolCall) -> PermissionDecision:
        # check_call 返回 allow，实际决策由 RuleMatcher 完成
        return "allow"


class ActPolicy:
    """act: 全部工具可见，写入和 shell 默认 ask。"""

    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tools

    def check_call(self, call: ToolCall) -> PermissionDecision:
        # check_call 返回 allow，实际决策由 RuleMatcher 完成
        return "allow"


_POLICIES: dict[ExecutionMode, ExecutionPolicy] = {
    "plan": PlanPolicy(),
    "build": BuildPolicy(),
    "act": ActPolicy(),
}


def parse_execution_mode(value: object) -> ExecutionMode | None:
    if not isinstance(value, str):
        return None
    match value:
        case "plan":
            return "plan"
        case "build":
            return "build"
        case "act":
            return "act"
        case _:
            return None


def policy_for_mode(mode: ExecutionMode) -> ExecutionPolicy:
    return _POLICIES[mode]


def registry_for_mode(
    registry: tuple[ToolSpec, ...],
    mode: ExecutionMode,
) -> tuple[ToolSpec, ...]:
    return policy_for_mode(mode).filter_tools(registry)


def mode_notice(mode: ExecutionMode) -> str:
    if mode == "plan":
        return (
            '<execution-mode name="plan">\n'
            "Plan Mode is active. Inspect and produce an action plan only. "
            "Do not modify code or run shell commands. You may create or update "
            "plan notes under .xcode/plans/*.md.\n"
            "</execution-mode>"
        )
    if mode == "build":
        return (
            '<execution-mode name="build">\n'
            "Build Mode is active. All tools are enabled; file writes and "
            "shell commands are allowed without HITL approval.\n"
            "</execution-mode>"
        )
    if mode == "act":
        return (
            '<execution-mode name="act">\n'
            "Act Mode is active. Read tools run directly; writes and shell "
            "commands require HITL approval unless configured otherwise.\n"
            "</execution-mode>"
        )
    return ""


# ── 初始化 MODE_DEFAULT_RULES ──


def _init_mode_rulesets() -> None:
    """填充 MODE_DEFAULT_RULES 全局字典。

    在模块导入时执行一次。build 默认只声明工具级权限，不内置命令分类。
    """
    if MODE_DEFAULT_RULES:
        return

    read_rules = (
        Rule(action="read_file", effect="allow"),
        Rule(action="glob_files", effect="allow"),
        Rule(action="grep_search", effect="allow"),
        Rule(action="find_files", effect="allow"),
        Rule(action="list_dir", effect="allow"),
        Rule(action="search_tools", effect="allow"),
        Rule(action="webfetch", effect="allow"),
        Rule(action="websearch", effect="allow"),
        Rule(action="question", effect="allow"),
        Rule(action="todowrite", effect="allow"),
        Rule(action="load_skill", effect="allow"),
    )
    write_rules = (
        Rule(action="write_file", effect="allow"),
        Rule(action="edit_file", effect="allow"),
        Rule(action="apply_patch", effect="allow"),
    )
    shell_rules = (
        Rule(action="bash", effect="allow"),
        Rule(action="shell", effect="allow"),
    )

    ask_write_rules = tuple(
        Rule(action=rule.action, effect="ask") for rule in write_rules
    )
    ask_shell_rules = tuple(
        Rule(action=rule.action, effect="ask") for rule in shell_rules
    )

    MODE_DEFAULT_RULES["plan"] = read_rules + (
        Rule(
            action="write_file",
            effect="allow",
            resource_pattern=".xcode/plans/*.md",
        ),
        Rule(
            action="edit_file",
            effect="allow",
            resource_pattern=".xcode/plans/*.md",
        ),
    )
    MODE_DEFAULT_RULES["build"] = read_rules + write_rules + shell_rules
    # Act 以 ask 为兜底；显式放行只读工具，其他工具需 HITL。
    MODE_DEFAULT_RULES["act"] = read_rules + ask_write_rules + ask_shell_rules


_init_mode_rulesets()
