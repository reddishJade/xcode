"""Shell 命令的 tree-sitter AST 语义分析。

职责：
- 对 bash/sh 和 PowerShell 命令做 tree-sitter 语法解析
- 提取可静态确认的路径参数（区分 read/write/delete）
- 标记不可静态确认的效果（变量/glob/命令替换/wrapper/解析错误）

依赖关系：
- 单向依赖 permission_model.py（Target, PermissionAccess, UnresolvedEffect）
- 不反向依赖 permission_model 的其他分组
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .permission_model import (
    PermissionAccess,
    Target,
    UnresolvedEffect,
    UnresolvedReason,
)
from .permission_model import Action, Constraint, PolicyEvaluator

logger = logging.getLogger(__name__)


# ── AST 不可用时的静默降级 ──

_HAS_TREE_SITTER = False
_HAS_BASH = False
_HAS_PWSH = False

try:
    from tree_sitter import Language, Parser, Tree  # noqa: F401

    _HAS_TREE_SITTER = True
except ImportError:
    logger.info("tree-sitter not available; shell AST analysis disabled")

if _HAS_TREE_SITTER:
    try:
        import tree_sitter_bash as tsbash

        _BASH_LANGUAGE = Language(tsbash.language())
        _HAS_BASH = True
    except Exception as exc:
        logger.warning("tree-sitter-bash not available: %s", exc)
        _BASH_LANGUAGE = None

    try:
        import tree_sitter_pwsh as tspwsh

        _PWSH_LANGUAGE = Language(tspwsh.language())
        _HAS_PWSH = True
    except Exception as exc:
        logger.warning("tree-sitter-pwsh not available: %s", exc)
        _PWSH_LANGUAGE = None


# ── 输出模型 ──


@dataclass(frozen=True)
class FileEffect:
    """命令对其参数的预期文件效果，用于 CommandSemanticsRegistry 注册。"""

    path: str
    access: PermissionAccess


@dataclass(frozen=True)
class ShellAnalysis:
    """一条 shell 命令的完整 AST 语义分析结果。"""

    resolved_paths: tuple[Target, ...]
    """可静态确认的路径参数，每个 Target 的 provenance="shell_literal" 已设置。"""
    unresolved_effects: tuple[UnresolvedEffect, ...]
    """不能静态确认的文件效果。"""
    primary_command: str | None
    """主命令名（cat/rm/git/...），小写。"""
    shell_type: Literal["posix", "powershell"]
    parse_error: bool
    """tree-sitter 是否返回 ERROR 节点。"""
    ast_available: bool
    """tree-sitter 是否成功加载。"""


# ── 命令语义注册表 ──


class CommandSemanticsRegistry:
    """将命令名 + 参数 → 文件效果的注册表。

    支持 posix（bash/sh/zsh/fish）和 powershell 两种语法。
    """

    def __init__(self) -> None:
        self._posix: dict[str, list[tuple[int, _ArgHandler]]] = {}
        self._pwsh: dict[str, list[tuple[int, _ArgHandler]]] = {}

    def register(
        self,
        command: str,
        syntax: Literal["posix", "powershell"] = "posix",
        priority: int = 0,
    ) -> callable:
        """装饰器注册命令语义处理器。

        priority 用于处理同名命令不同语法的优先级（越高越优先）。
        """
        target = self._posix if syntax == "posix" else self._pwsh

        def decorator(fn: _ArgHandler) -> _ArgHandler:
            handlers = target.setdefault(command, [])
            handlers.append((priority, fn))
            handlers.sort(key=lambda x: -x[0])
            return fn

        return decorator

    def handle(
        self,
        command: str,
        argv: list[str],
        syntax: Literal["posix", "powershell"],
    ) -> list[FileEffect | UnresolvedEffect]:
        """对给定命令和参数返回文件效果。"""
        target = self._posix if syntax == "posix" else self._pwsh
        handlers = target.get(command.lower(), [])
        if not handlers:
            # 未知命令 → 保守标记为 unresolved wrapper_command
            return [
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment=f"unknown command: {command}",
                )
            ]
        # 取优先级最高的处理器
        _, handler = handlers[0]
        return list(handler(argv))


type _ArgHandler = callable[[list[str]], list[FileEffect | UnresolvedEffect]]


# ── 内置命令注册 ──

_REGISTRY = CommandSemanticsRegistry()


@_REGISTRY.register("cat", syntax="posix")
def _cat(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return [_effect_from_arg(arg, "read") for arg in argv[1:] if _is_path_like(arg)]


@_REGISTRY.register("head", syntax="posix")
@_REGISTRY.register("tail", syntax="posix")
@_REGISTRY.register("less", syntax="posix")
@_REGISTRY.register("more", syntax="posix")
def _read_file_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """head/tail/less/more [options] [file...]

    跳过短选项及其值（如 -n 5），只收集路径参数。
    """
    effects: list[FileEffect | UnresolvedEffect] = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-"):
            if _is_short_option_with_value(arg):
                skip_next = True
            continue
        if _is_path_like(arg):
            effects.append(_effect_from_arg(arg, "read"))
    return effects


@_REGISTRY.register("ls", syntax="posix")
@_REGISTRY.register("dir", syntax="posix")
@_REGISTRY.register("realpath", syntax="posix")
def _list_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return [_effect_from_arg(arg, "read") for arg in argv[1:] if _is_path_like(arg)]


@_REGISTRY.register("cp", syntax="posix")
def _cp(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    effects: list[FileEffect | UnresolvedEffect] = []
    if len(args) >= 2:
        # cp src dst → src=read, dst=write
        effects.append(_effect_from_arg(args[-2], "read"))
        effects.append(_effect_from_arg(args[-1], "write"))
    elif len(args) == 1 and args[0].startswith("-"):
        pass  # flag-only
    return effects


@_REGISTRY.register("mv", syntax="posix")
def _mv(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    effects: list[FileEffect | UnresolvedEffect] = []
    if len(args) >= 2:
        effects.append(_effect_from_arg(args[-2], "write"))
        effects.append(_effect_from_arg(args[-1], "write"))
    return effects


@_REGISTRY.register("rm", syntax="posix")
def _rm(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(arg, "delete") for arg in args if _is_path_like(arg)]


@_REGISTRY.register("echo", syntax="posix")
def _echo(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # echo 本身只写 stdout，不操作文件
    return []


@_REGISTRY.register("mkdir", syntax="posix")
@_REGISTRY.register("touch", syntax="posix")
def _create_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(arg, "write") for arg in args if _is_path_like(arg)]


@_REGISTRY.register("chmod", syntax="posix")
@_REGISTRY.register("chown", syntax="posix")
def _perm_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(arg, "write") for arg in args if _is_path_like(arg)]


@_REGISTRY.register("grep", syntax="posix")
@_REGISTRY.register("rg", syntax="posix")
@_REGISTRY.register("ack", syntax="posix")
def _grep_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    # grep pattern [file...] — 跳过第一个非 flag 参数（pattern）
    files = args[1:] if len(args) > 1 else []
    return [_effect_from_arg(arg, "read") for arg in files if _is_path_like(arg)]


@_REGISTRY.register("diff", syntax="posix")
@_REGISTRY.register("cmp", syntax="posix")
def _diff_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(arg, "read") for arg in args if _is_path_like(arg)]


@_REGISTRY.register("git", syntax="posix")
def _git(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # git 大多数子命令读 .git 目录；写操作 git reset --hard 会被 SafetyBackstop 处理
    return []


@_REGISTRY.register("cd", syntax="posix")
@_REGISTRY.register("pushd", syntax="posix")
@_REGISTRY.register("popd", syntax="posix")
def _cd_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # cd/pushd/popd 改变工作目录，不直接产生文件读/写效果
    return []


@_REGISTRY.register("find", syntax="posix")
def _find(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    effects: list[FileEffect | UnresolvedEffect] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-exec" and i + 1 < len(argv):
            effects.append(
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment=f"find -exec {argv[i+1]}: paths from find output",
                )
            )
            break  # 不再继续解析
        elif arg == "-ok":
            effects.append(
                UnresolvedEffect(reason="wrapper_command", fragment="find -ok")
            )
            break
        elif arg == "-delete":
            effects.append(
                UnresolvedEffect(reason="wrapper_command", fragment="find -delete")
            )
            break
        i += 1
    return effects


@_REGISTRY.register("xargs", syntax="posix")
def _xargs(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # xargs 不带命令时默认 echo（安全）
    # 带命令时如 xargs rm，内层命令路径来自 stdin（不可静态分析）
    args = [a for a in argv[1:] if not _is_flag(a)]
    if args:
        return [
            UnresolvedEffect(
                reason="wrapper_command",
                fragment=f"xargs {args[0]}: args from stdin",
            )
        ]
    return []


@_REGISTRY.register("bash", syntax="posix")
@_REGISTRY.register("sh", syntax="posix")
@_REGISTRY.register("zsh", syntax="posix")
def _shell_wrapper(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # bash -c "dangerous command" → 无法静态分析内联命令
    args = [a for a in argv[1:] if not _is_flag(a)]
    if args:
        return [
            UnresolvedEffect(
                reason="eval_like",
                fragment=f"{argv[0]} -c: dynamic command execution",
            )
        ]
    return []


@_REGISTRY.register("source", syntax="posix")
@_REGISTRY.register(".", syntax="posix")
def _source_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    args = [a for a in argv[1:] if not _is_flag(a)]
    if args:
        return [
            UnresolvedEffect(
                reason="eval_like",
                fragment=f"source {args[0]}: dynamic script execution",
            )
        ]
    return []


@_REGISTRY.register("eval", syntax="posix")
def _eval_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return [
        UnresolvedEffect(
            reason="eval_like", fragment="eval: fully dynamic execution"
        )
    ]


# ── PowerShell 命令注册 ──


@_REGISTRY.register("echo", syntax="powershell")
@_REGISTRY.register("write-output", syntax="powershell")
@_REGISTRY.register("write-host", syntax="powershell")
def _ps_noop(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """PowerShell 中无文件效果的常见命令。"""
    return []


@_REGISTRY.register("get-content", syntax="powershell")
@_REGISTRY.register("gc", syntax="powershell")
@_REGISTRY.register("type", syntax="powershell")
@_REGISTRY.register("cat", syntax="powershell")
def _ps_read(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return _ps_path_param_effects(argv, "read")


@_REGISTRY.register("set-content", syntax="powershell")
@_REGISTRY.register("sc", syntax="powershell")
@_REGISTRY.register("out-file", syntax="powershell")
def _ps_write(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return _ps_path_param_effects(argv, "write")


@_REGISTRY.register("add-content", syntax="powershell")
@_REGISTRY.register("ac", syntax="powershell")
def _ps_append(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return _ps_path_param_effects(argv, "write")


@_REGISTRY.register("remove-item", syntax="powershell")
@_REGISTRY.register("del", syntax="powershell")
@_REGISTRY.register("ri", syntax="powershell")
def _ps_delete(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return _ps_path_param_effects(argv, "delete")


@_REGISTRY.register("copy-item", syntax="powershell")
@_REGISTRY.register("copy", syntax="powershell")
@_REGISTRY.register("cp", syntax="powershell")
def _ps_copy(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # copy-item src dst → src=read, dst=write
    paths = _ps_extract_paths(argv)
    effects: list[FileEffect | UnresolvedEffect] = []
    if len(paths) >= 2:
        effects.append(_effect_from_arg(paths[-2], "read"))
        effects.append(_effect_from_arg(paths[-1], "write"))
    return effects


@_REGISTRY.register("move-item", syntax="powershell")
@_REGISTRY.register("move", syntax="powershell")
@_REGISTRY.register("mv", syntax="powershell")
def _ps_move(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    paths = _ps_extract_paths(argv)
    effects: list[FileEffect | UnresolvedEffect] = []
    if len(paths) >= 2:
        effects.append(_effect_from_arg(paths[-2], "write"))
        effects.append(_effect_from_arg(paths[-1], "write"))
    return effects


@_REGISTRY.register("get-childitem", syntax="powershell")
@_REGISTRY.register("gci", syntax="powershell")
@_REGISTRY.register("ls", syntax="powershell")
@_REGISTRY.register("dir", syntax="powershell")
def _ps_list(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    return _ps_path_param_effects(argv, "read")


@_REGISTRY.register("select-string", syntax="powershell")
@_REGISTRY.register("sls", syntax="powershell")
def _ps_select_string(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    # select-string pattern [-Path files]
    has_path = False
    effects: list[FileEffect | UnresolvedEffect] = []
    for i, arg in enumerate(argv):
        if arg.lower() in ("-path", "-literalpath"):
            if i + 1 < len(argv):
                effects.append(_effect_from_arg(argv[i + 1], "read"))
                has_path = True
    if not has_path:
        # 管道输入，无法静态分析
        pass
    return effects


# ── POSIX AST 分析器 ──


class PosixAnalyzer:
    """使用 tree-sitter-bash 解析 POSIX shell 命令。"""

    def __init__(self, registry: CommandSemanticsRegistry | None = None) -> None:
        self._registry = registry or _REGISTRY
        self._parser: Parser | None = None
        if _HAS_BASH and _HAS_TREE_SITTER:
            self._parser = Parser(_BASH_LANGUAGE)

    def analyze(self, command: str) -> ShellAnalysis:
        if self._parser is None:
            return ShellAnalysis(
                resolved_paths=(),
                unresolved_effects=(
                    UnresolvedEffect(
                        reason="unsupported_shell",
                        fragment="bash AST not available",
                    ),
                ),
                primary_command=_guess_primary_command(command),
                shell_type="posix",
                parse_error=True,
                ast_available=False,
            )

        text = command.encode("utf-8")
        tree = self._parser.parse(text)
        root = tree.root_node

        has_error = root.has_error
        resolved_paths: list[Target] = []
        unresolved: list[UnresolvedEffect] = []
        seen_args: set[str] = set()

        # 遍历所有命令节点（包括管道、子 shell 内嵌等）
        self._walk_nodes(command, text, root, resolved_paths, unresolved, seen_args)

        primary = _guess_primary_command(command)

        return ShellAnalysis(
            resolved_paths=tuple(resolved_paths),
            unresolved_effects=tuple(unresolved),
            primary_command=primary,
            shell_type="posix",
            parse_error=has_error,
            ast_available=True,
        )

    def _walk_nodes(
        self,
        command: str,
        text: bytes,
        node: object,
        resolved: list[Target],
        unresolved: list[UnresolvedEffect],
        seen_args: set[str],
    ) -> None:
        """递归遍历 AST 节点。"""
        # 处理命令节点
        if node.type == "command":
            self._handle_command_node(command, text, node, resolved, unresolved, seen_args)
            return  # 不递归进子命令（command 的 child 是 word，非子命令）

        if node.type == "redirected_statement":
            # 遍历子节点：command + file_redirect
            for child in node.children:
                if child.type == "command":
                    self._handle_command_node(
                        command, text, child, resolved, unresolved, seen_args
                    )
                elif child.type in ("file_redirect", "herestring_redirect", "heredoc_redirect"):
                    self._handle_redirect_node(text, child, resolved, unresolved, seen_args)
            return

        if node.type in ("file_redirect", "herestring_redirect", "heredoc_redirect"):
            self._handle_redirect_node(text, node, resolved, unresolved, seen_args)
            return

        if node.type == "list":
            for child in node.children:
                self._walk_nodes(command, text, child, resolved, unresolved, seen_args)
            return

        if node.type == "pipeline":
            for child in node.children:
                self._walk_nodes(command, text, child, resolved, unresolved, seen_args)
            return

        # 子 shell 和命令替换 → 标记 unresolved
        if node.type == "subshell":
            unresolved.append(
                UnresolvedEffect(reason="command_substitution", fragment="(...) subshell")
            )
            # 仍然递归分析内部命令
            for child in node.children:
                self._walk_nodes(command, text, child, resolved, unresolved, seen_args)
            return

        if node.type == "command_substitution":
            unresolved.append(
                UnresolvedEffect(
                    reason="command_substitution",
                    fragment=text[node.start_byte : node.end_byte].decode(),
                )
            )
            return

        # ERROR 节点 → 解析失败
        if node.type == "ERROR":
            unresolved.append(
                UnresolvedEffect(
                    reason="parse_error",
                    fragment=text[node.start_byte : node.end_byte].decode(),
                )
            )
            return

        # 递归其他节点
        for child in node.children:
            self._walk_nodes(command, text, child, resolved, unresolved, seen_args)

    def _handle_command_node(
        self,
        command: str,
        text: bytes,
        node: object,
        resolved: list[Target],
        unresolved: list[UnresolvedEffect],
        seen_args: set[str],
    ) -> None:
        """处理一个 command AST 节点。"""
        # tree-sitter-bash 中 command 的命名子节点 field 名为 "name"
        cmd_name_node = _child_by_field(node, "name")
        if cmd_name_node is None:
            # fallback: command_name 是 command 的第一个子节点
            for child in node.children:
                if child.type == "command_name":
                    cmd_name_node = child
                    break

        if cmd_name_node is None:
            unresolved.append(
                UnresolvedEffect(reason="parse_error", fragment="no command name in AST")
            )
            return

        # command_name 内部可能有 word/word_with_tilde_expansion 等
        cmd_name_text_node = _first_name_word(cmd_name_node)
        if cmd_name_text_node is None:
            unresolved.append(
                UnresolvedEffect(reason="parse_error", fragment="empty command_name in AST")
            )
            return

        cmd_name = text[cmd_name_text_node.start_byte : cmd_name_text_node.end_byte].decode().strip().lower()

        # 构建 argv：命令名本身 + 同级的 word/string 参数
        argv = [cmd_name]
        has_expansion = False
        has_glob = False

        for child in node.children:
            if child is cmd_name_node:
                continue  # 跳过命令名节点本身
            if child.type == "word":
                val = text[child.start_byte : child.end_byte].decode()
                argv.append(val)
            elif child.type == "string":
                val = text[child.start_byte : child.end_byte].decode()
                argv.append(val)
            elif child.type == "concatenation":
                has_expansion = True
                val = text[child.start_byte : child.end_byte].decode()
                argv.append(val)
            elif child.type in ("simple_expansion", "expansion"):
                has_expansion = True
                val = text[child.start_byte : child.end_byte].decode()
                argv.append(val)
            elif child.type in ("number",):
                val = text[child.start_byte : child.end_byte].decode()
                argv.append(val)
            elif child.type == "command_substitution":
                # $(...) 内的命令无法静态分析路径，不加入 argv（避免注册表二次处理）
                sub_text = text[child.start_byte : child.end_byte].decode()
                unresolved.append(
                    UnresolvedEffect(
                        reason="command_substitution",
                        fragment=sub_text[:80],
                    )
                )
            elif child.type == "subshell":
                sub_text = text[child.start_byte : child.end_byte].decode()
                unresolved.append(
                    UnresolvedEffect(
                        reason="command_substitution",
                        fragment=f"(subshell): {sub_text[:60]}",
                    )
                )

        # 检查 glob 模式
        for a in argv[1:]:
            stripped = a.strip("\"'")
            if _looks_like_glob(stripped):
                has_glob = True

        # 通过语义注册表解析
        effects = self._registry.handle(cmd_name, argv, "posix")

        for effect in effects:
            if isinstance(effect, UnresolvedEffect):
                unresolved.append(effect)
            elif isinstance(effect, FileEffect):
                if has_expansion:
                    unresolved.append(
                        UnresolvedEffect(
                            reason="variable_expansion",
                            fragment=effect.path,
                        )
                    )
                elif has_glob:
                    unresolved.append(
                        UnresolvedEffect(reason="glob", fragment=effect.path)
                    )
                elif effect.path not in seen_args:
                    seen_args.add(effect.path)
                    resolved.append(
                        Target(
                            kind="path",
                            value=effect.path,
                            access=effect.access,
                            provenance="shell_literal",
                        )
                    )

        # SafetyBackstop 的风格标记：路径含 .. 或绝对路径的检测在
        # PathBoundaryPolicyEvaluator 中进行，这里不做重复工作

    def _handle_redirect_node(
        self,
        text: bytes,
        node: object,
        resolved: list[Target],
        unresolved: list[UnresolvedEffect],
        seen_args: set[str],
    ) -> None:
        """处理重定向节点（> file, >> file, < file 等）。"""
        for child in node.children:
            if child.type in ("word", "string"):
                path = text[child.start_byte : child.end_byte].decode().strip("\"'")
                if path:
                    # '>' 重定向是写，'<' 是读
                    access: PermissionAccess = "write"
                    # 检查是否有简单展开
                    has_var = "$" in path or "~" in path
                    if has_var:
                        unresolved.append(
                            UnresolvedEffect(
                                reason="variable_expansion",
                                fragment=f"redirect: {path}",
                            )
                        )
                    elif path not in seen_args:
                        seen_args.add(path)
                        resolved.append(
                            Target(
                                kind="path",
                                value=path,
                                access=access,
                                provenance="shell_literal",
                            )
                        )


# ── PowerShell AST 分析器 ──


class PowerShellAnalyzer:
    """使用 tree-sitter-pwsh 解析 PowerShell 命令。"""

    def __init__(self, registry: CommandSemanticsRegistry | None = None) -> None:
        self._registry = registry or _REGISTRY
        self._parser: Parser | None = None
        if _HAS_PWSH and _HAS_TREE_SITTER:
            self._parser = Parser(_PWSH_LANGUAGE)

    def analyze(self, command: str) -> ShellAnalysis:
        if self._parser is None:
            return ShellAnalysis(
                resolved_paths=(),
                unresolved_effects=(
                    UnresolvedEffect(
                        reason="unsupported_shell",
                        fragment="PowerShell AST not available",
                    ),
                ),
                primary_command=_guess_primary_command(command),
                shell_type="powershell",
                parse_error=True,
                ast_available=False,
            )

        text = command.encode("utf-8")
        tree = self._parser.parse(text)
        root = tree.root_node

        has_error = root.has_error
        resolved_paths: list[Target] = []
        unresolved: list[UnresolvedEffect] = []
        seen_args: set[str] = set()

        self._walk_nodes(command, text, root, resolved_paths, unresolved, seen_args)

        return ShellAnalysis(
            resolved_paths=tuple(resolved_paths),
            unresolved_effects=tuple(unresolved),
            primary_command=_guess_primary_command(command),
            shell_type="powershell",
            parse_error=has_error,
            ast_available=True,
        )

    def _walk_nodes(
        self,
        command: str,
        text: bytes,
        node: object,
        resolved: list[Target],
        unresolved: list[UnresolvedEffect],
        seen_args: set[str],
    ) -> None:
        if node.type == "command":
            self._handle_command_node(command, text, node, resolved, unresolved, seen_args)
            return

        if node.type == "pipeline":
            for child in node.children:
                self._walk_nodes(command, text, child, resolved, unresolved, seen_args)
            return

        if node.type == "pipeline_chain":
            for child in node.children:
                self._walk_nodes(command, text, child, resolved, unresolved, seen_args)
            return

        if node.type in ("redirect", "redirection"):
            self._handle_redirect_node(text, node, resolved, unresolved, seen_args)
            return

        # ERROR node
        if node.type == "ERROR":
            unresolved.append(
                UnresolvedEffect(
                    reason="parse_error",
                    fragment=text[node.start_byte : node.end_byte].decode(),
                )
            )
            return

        # Recursive walk
        for child in node.children:
            self._walk_nodes(command, text, child, resolved, unresolved, seen_args)

    def _handle_command_node(
        self,
        command: str,
        text: bytes,
        node: object,
        resolved: list[Target],
        unresolved: list[UnresolvedEffect],
        seen_args: set[str],
    ) -> None:
        # Extract command name
        cmd_name_node = _child_by_field(node, "command_name")
        if cmd_name_node is None:
            return

        cmd_name = text[cmd_name_node.start_byte : cmd_name_node.end_byte].decode().strip().lower()

        # Extract all arguments from command_elements
        argv = [cmd_name]
        has_variable = False

        # Find command_elements
        for child in node.children:
            if child.type == "command_elements":
                for elem in child.children:
                    if elem.type == "generic_token":
                        val = text[elem.start_byte : elem.end_byte].decode()
                        argv.append(val)
                    elif elem.type == "command_parameter":
                        val = text[elem.start_byte : elem.end_byte].decode()
                        argv.append(val)
                    elif elem.type == "unary_expression":
                        val = text[elem.start_byte : elem.end_byte].decode()
                        argv.append(val)
                        # 字符串字面量可能有插值
                        if "$" in val:
                            has_variable = True
                    elif elem.type in (
                        "variable_access",
                        "variable_access_expression",
                    ):
                        has_variable = True
                        val = text[elem.start_byte : elem.end_byte].decode()
                        argv.append(val)

        effects = self._registry.handle(cmd_name, argv, "powershell")

        for effect in effects:
            if isinstance(effect, UnresolvedEffect):
                unresolved.append(effect)
            elif isinstance(effect, FileEffect):
                if has_variable:
                    unresolved.append(
                        UnresolvedEffect(
                            reason="variable_expansion",
                            fragment=effect.path,
                        )
                    )
                elif effect.path not in seen_args:
                    seen_args.add(effect.path)
                    resolved.append(
                        Target(
                            kind="path",
                            value=effect.path,
                            access=effect.access,
                            provenance="shell_literal",
                        )
                    )

    def _handle_redirect_node(
        self,
        text: bytes,
        node: object,
        resolved: list[Target],
        unresolved: list[UnresolvedEffect],
        seen_args: set[str],
    ) -> None:
        # PowerShell redirection: > file, >> file, 2>&1, etc.
        for child in node.children:
            if child.type in ("generic_token", "string_literal_expression"):
                path_val = text[child.start_byte : child.end_byte].decode()
                path = path_val.strip("\"'")
                if path and path not in seen_args:
                    has_var = "$" in path
                    if has_var:
                        unresolved.append(
                            UnresolvedEffect(
                                reason="variable_expansion",
                                fragment=f"redirect: {path}",
                            )
                        )
                    else:
                        seen_args.add(path)
                        resolved.append(
                            Target(
                                kind="path",
                                value=path,
                                access="write",
                                provenance="shell_literal",
                            )
                        )


# ── 统一入口 ──


class ShellAnalysisPolicyEvaluator:
    """将 shell 命令中不可静态确认的效果转为 ask 约束。

    此 evaluator 读取 action.unresolved_effects（由 ActionExtractor
    或 ShellAnalyzer 负责填充），为每个 unresolved 效果生成 ask 约束。

    与 SafetyBackstopPolicyEvaluator 的关系：
    - SafetyBackstop 做命令级三桶分类（命令名模式匹配）
    - ShellAnalysisPolicyEvaluator 做路径级 AST 语义分析
    - 两者是互补的，SafetyBackstop 的 deny 会覆盖任何 ask
    """

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        if not action.unresolved_effects:
            return ()

        constraints: list[Constraint] = []
        for effect in action.unresolved_effects:
            constraints.append(
                Constraint(
                    decision="ask",
                    source="shell_analysis",
                    reason=(
                        f"unresolved shell effect: {effect.reason}"
                        f" - {effect.fragment}"
                    ),
                )
            )
        return tuple(constraints)


def analyze_shell_command(
    command: str,
    shell_type: str = "posix",
) -> ShellAnalysis:
    """统一入口：按 shell 类型选择解析器。

    Args:
        command: shell 命令文本。
        shell_type: "posix"（bash/zsh/sh/fish）或 "powershell"。

    Returns:
        ShellAnalysis 包含可确认路径和不可确认效果。
    """
    if shell_type == "powershell":
        analyzer = PowerShellAnalyzer()
    else:
        analyzer = PosixAnalyzer()
    return analyzer.analyze(command)


# ── 辅助函数 ──


def _child_by_field(node: object, field_name: str) -> object | None:
    """获取 tree-sitter 节点的命名子节点（field child）。

    tree-sitter 0.25+ 的 Node 对象通常有 child_by_field_name 方法。
    """
    if hasattr(node, "child_by_field_name"):
        return node.child_by_field_name(field_name)
    return None


def _first_name_word(node: object) -> object | None:
    """从 command_name 或类似节点中提取首个子 word 节点。

    command_name 的 child 可能是 word、word_with_tilde_expansion、
    或带扩展的 concatenation。取第一个非展开的简单 word。
    """
    if not hasattr(node, "children"):
        return None
    for child in node.children:
        if child.type in ("word",):
            return child
    # 如果没有简单 word，取第一个子节点
    if node.children:
        return node.children[0]
    return None


def _is_path_like(arg: str) -> bool:
    """粗略判断参数是否可能是路径（非 flag、非纯选项值）。"""
    stripped = arg.strip("\"'")
    if not stripped:
        return False
    if stripped.startswith("-"):
        return False
    # 只含操作符/分隔符的排除
    if stripped in {"&&", "||", ";", "|", "(", ")", "{", "}"}:
        return False
    return True


def _is_flag(arg: str) -> bool:
    return arg.startswith("-")


def _is_short_option_with_value(arg: str) -> bool:
    """检查短选项是否期望一个值参数（如 -n 5, -f file）。

    假设单字母短选项后面有值，长选项（--name VALUE）另作处理。
    """
    if arg.startswith("--"):
        # 长选项通常有 = 分隔，或单独处理
        return "=" not in arg
    # 单破折号 + 单字母短选项（-n, -f 等）
    return len(arg) == 2 and arg[0] == "-" and arg[1].isalpha()


def _looks_like_glob(s: str) -> bool:
    """检查字符串是否包含 glob 通配符。"""
    return "*" in s or "?" in s or ("[" in s and "]" in s)


def _effect_from_arg(
    arg: str, access: PermissionAccess
) -> FileEffect | UnresolvedEffect:
    """将参数转为 FileEffect，如果含变量/glob 则转为 UnresolvedEffect。"""
    stripped = arg.strip("\"'")
    if "$" in stripped:
        return UnresolvedEffect(reason="variable_expansion", fragment=stripped)
    if _looks_like_glob(stripped):
        return UnresolvedEffect(reason="glob", fragment=stripped)
    return FileEffect(path=stripped, access=access)


def _guess_primary_command(command: str) -> str | None:
    """粗略提取命令字符串的首个命令名（无 AST 时的 fallback）。"""
    stripped = command.strip()
    if not stripped:
        return None
    # 跳过开头的变量赋值
    parts = stripped.split()
    for part in parts:
        if "=" in part and not part.startswith("-"):
            continue
        return part.split("/")[-1].lower()
    return None


def _ps_extract_paths(argv: list[str]) -> list[str]:
    """从 PowerShell 参数列表中提取路径参数（跳过 -Path 等标志）。"""
    paths: list[str] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("-"):
            # -Path value
            if i + 1 < len(argv):
                # 跳过 -Flag value（但 value 可能是路径）
                val = argv[i + 1]
                if not val.startswith("-"):
                    # 标记为路径
                    paths.append(val.strip("\"'"))
                    i += 2
                    continue
            i += 1
        else:
            paths.append(arg.strip("\"'"))
            i += 1
    return paths


def _ps_path_param_effects(
    argv: list[str], access: PermissionAccess
) -> list[FileEffect | UnresolvedEffect]:
    """从 PowerShell 命令提取 -Path/-LiteralPath 参数或位置路径参数。"""
    effects: list[FileEffect | UnresolvedEffect] = []
    found = False
    for i, arg in enumerate(argv):
        if arg.lower() in ("-path", "-literalpath", "-literal_path"):
            if i + 1 < len(argv):
                effects.append(_effect_from_arg(argv[i + 1], access))
                found = True
        elif arg.lower() == "-value":
            continue
    if not found:
        # 位置参数
        paths = _ps_extract_paths(argv)
        effects.extend(_effect_from_arg(p, access) for p in paths)
    return effects
