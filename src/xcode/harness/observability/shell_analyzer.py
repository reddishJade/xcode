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
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal

from .permission_model import (
    PermissionAccess,
    Target,
    UnresolvedEffect,
)
from .permission_model import Action, Constraint

logger = logging.getLogger(__name__)


# ── AST 不可用时的静默降级 ──

_HAS_TREE_SITTER = False
_HAS_BASH = False
_HAS_PWSH = False

Language: Any = None
Parser: Any = None
Tree: Any = None

try:
    from tree_sitter import Language as _Language, Parser as _Parser, Tree as _Tree  # noqa: F401

    Language = _Language
    Parser = _Parser
    Tree = _Tree
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
    shell_type: Literal["posix", "powershell", "cmd"]
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
    ) -> Callable:
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


type _ArgHandler = Callable[[list[str]], list[FileEffect | UnresolvedEffect]]


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


# ── 扩展 POSIX 命令注册 ──


@_REGISTRY.register("tee", syntax="posix")
def _tee(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """tee [options] [file...] — 写入文件。"""
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(a, "write") for a in args if _is_path_like(a)]


@_REGISTRY.register("wc", syntax="posix")
@_REGISTRY.register("sort", syntax="posix")
@_REGISTRY.register("uniq", syntax="posix")
@_REGISTRY.register("cut", syntax="posix")
@_REGISTRY.register("comm", syntax="posix")
def _read_file_simple(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """wc/sort/uniq/cut/comm [options] [file...] — 读文件。

    跳过 -- 和短选项；对短选项不自动跳过下一参数，
    因为 -l、-w、-c 等不取值。只收集非 flag 路径参数。
    """
    effects: list[FileEffect | UnresolvedEffect] = []
    for arg in argv[1:]:
        if arg == "--":
            continue
        if arg.startswith("-"):
            continue
        if _is_path_like(arg):
            effects.append(_effect_from_arg(arg, "read"))
    return effects


@_REGISTRY.register("diff3", syntax="posix")
def _two_file_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """diff3 file1 file2 file3 — 比较三个文件（只读）。"""
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(a, "read") for a in args if _is_path_like(a)]


@_REGISTRY.register("du", syntax="posix")
@_REGISTRY.register("df", syntax="posix")
def _disk_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """du/df [path...] — 磁盘使用（只读）。"""
    args = [a for a in argv[1:] if not _is_flag(a)]
    return [_effect_from_arg(a, "read") for a in args if _is_path_like(a)]


@_REGISTRY.register("curl", syntax="posix")
def _curl(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """curl [options] [URL...] — 下载文件。

    -o/--output file → write
    -O/--remote-name → write (文件名从 URL 推断，不可静态分析)
    """
    effects: list[FileEffect | UnresolvedEffect] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-o", "--output") and i + 1 < len(argv):
            effects.append(_effect_from_arg(argv[i + 1], "write"))
            i += 2
            continue
        if arg in ("-O", "--remote-name"):
            effects.append(
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment="curl -O: filename from URL",
                )
            )
            i += 1
            continue
        if arg.startswith("-"):
            i += 1
            continue
        # 非 flag 参数通常是 URL，不是文件路径
        i += 1
    return effects


@_REGISTRY.register("wget", syntax="posix")
def _wget(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """wget [options] [URL...] — 下载文件。

    -O/--output-document file → write
    """
    effects: list[FileEffect | UnresolvedEffect] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-O", "--output-document") and i + 1 < len(argv):
            effects.append(_effect_from_arg(argv[i + 1], "write"))
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        i += 1
    return effects


@_REGISTRY.register("sed", syntax="posix")
def _sed(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """sed [options] [script] [file...] — 流编辑器。

    -i[SUFFIX] → 原地修改文件（write）
    位置参数中的文件 → read（如果 -i 则为 write）
    """
    inplace = False
    files: list[str] = []
    seen_script = False  # sed 脚本已遇到
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-i"):
            inplace = True
            continue
        if arg == "--":
            skip_next = False
            continue
        if arg.startswith("-"):
            if _is_short_option_with_value(arg) and arg not in ("-e", "--expression"):
                skip_next = True
            continue
        if not seen_script and _is_sed_script(arg):
            seen_script = True
            continue
        if _is_path_like(arg):
            files.append(arg)
    if not files:
        return []
    access: PermissionAccess = "write" if inplace else "read"
    return [_effect_from_arg(f, access) for f in files]


@_REGISTRY.register("awk", syntax="posix")
def _awk(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """awk [options] [script] [file...] — 模式扫描（读文件）。"""
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


@_REGISTRY.register("tar", syntax="posix")
def _tar(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """tar [options] [file...] — 归档工具。

    创建模式（-c）：写入 archive file
    提取模式（-x）：读取 archive file，写入目标
    列出模式（-t）：只读
    """
    create = False
    extract = False
    list_mode = False
    archive_file: str | None = None
    args: list[str] = []
    pending_f = False  # 下一个非 flag 参数是 archive 文件名
    seen_action = False  # 已经处理过动作字母（c/x/t）

    def _parse_flags(chars: str) -> None:
        """解析 tar flag 字符串（可能来自 -czf 或裸 czf）。"""
        nonlocal create, extract, list_mode, pending_f, seen_action
        for ch in chars:
            if ch == "c":
                create = True
                seen_action = True
            elif ch == "x":
                extract = True
                seen_action = True
            elif ch == "t":
                list_mode = True
                seen_action = True
            elif ch == "f":
                pending_f = True

    for arg in argv[1:]:
        if pending_f:
            archive_file = arg
            pending_f = False
            continue
        if arg.startswith("-"):
            _parse_flags(arg.lstrip("-"))
            continue
        # GNU tar 允许裸标志（不带 -）：tar czf archive...
        if not seen_action and arg.isalpha():
            _parse_flags(arg)
            continue
        if archive_file is None and pending_f:
            archive_file = arg
            pending_f = False
            continue
        args.append(arg)

    effects: list[FileEffect | UnresolvedEffect] = []
    if archive_file:
        if create:
            effects.append(_effect_from_arg(archive_file, "write"))
        elif extract:
            effects.append(_effect_from_arg(archive_file, "read"))
            effects.append(
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment="tar -x: extracts to cwd",
                )
            )
        else:
            effects.append(_effect_from_arg(archive_file, "read"))
    if extract and not archive_file:
        # tar xf without explicit file (reads from stdin)
        effects.append(
            UnresolvedEffect(reason="wrapper_command", fragment="tar: extracts from stdin")
        )
    for a in args:
        effects.append(_effect_from_arg(a, "read"))
    return effects


@_REGISTRY.register("gzip", syntax="posix")
@_REGISTRY.register("gunzip", syntax="posix")
@_REGISTRY.register("bzip2", syntax="posix")
@_REGISTRY.register("bunzip2", syntax="posix")
@_REGISTRY.register("xz", syntax="posix")
@_REGISTRY.register("unxz", syntax="posix")
def _compress_cmd(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """gzip/gunzip/bzip2/bunzip2/xz/unxz [file...]

    压缩：read → write（原地替换为 .gz 文件）
    解压：read → delete（原地删除源文件）
    """
    args = [a for a in argv[1:] if not _is_flag(a)]
    effects: list[FileEffect | UnresolvedEffect] = []
    for a in args:
        if _is_path_like(a):
            effects.append(_effect_from_arg(a, "read"))
    return effects


@_REGISTRY.register("unzip", syntax="posix")
def _unzip(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """unzip [options] archive[.zip] [file...] — 解压。"""
    args = [a for a in argv[1:] if not _is_flag(a)]
    effects: list[FileEffect | UnresolvedEffect] = []
    if args:
        effects.append(_effect_from_arg(args[0], "read"))
        effects.append(
            UnresolvedEffect(
                reason="wrapper_command", fragment="unzip: extracts to cwd"
            )
        )
    return effects


@_REGISTRY.register("zip", syntax="posix")
def _zip(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """zip [options] archive[.zip] [file...] — 压缩。"""
    args = [a for a in argv[1:] if not _is_flag(a)]
    effects: list[FileEffect | UnresolvedEffect] = []
    if len(args) >= 1:
        effects.append(_effect_from_arg(args[0], "write"))
    for a in args[1:]:
        if _is_path_like(a):
            effects.append(_effect_from_arg(a, "read"))
    return effects


@_REGISTRY.register("pip", syntax="posix")
@_REGISTRY.register("pip3", syntax="posix")
def _pip(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """pip install/uninstall → 修改系统 Python 包（write）。"""
    if len(argv) > 1 and argv[1] in ("install", "uninstall", "download"):
        return [UnresolvedEffect(reason="wrapper_command", fragment="pip: modifies packages")]
    return []


@_REGISTRY.register("npm", syntax="posix")
def _npm(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """npm install/build → 修改 node_modules。"""
    if len(argv) > 1 and argv[1] in ("install", "ci", "add", "update", "audit fix"):
        return [UnresolvedEffect(reason="wrapper_command", fragment="npm: modifies node_modules")]
    if len(argv) > 1 and argv[1] in ("run", "exec"):
        return [UnresolvedEffect(reason="wrapper_command", fragment="npm run: dynamic execution")]
    return []


@_REGISTRY.register("npx", syntax="posix")
def _npx(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """npx [command] — 临时执行包（动态）。"""
    return [UnresolvedEffect(reason="wrapper_command", fragment="npx: dynamic execution")]


@_REGISTRY.register("make", syntax="posix")
def _make(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """make [target...] — 运行构建。"""
    return [UnresolvedEffect(reason="wrapper_command", fragment="make: build tool")]


@_REGISTRY.register("cargo", syntax="posix")
def _cargo(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """cargo build/test → 构建 Rust 项目。"""
    if len(argv) > 1 and argv[1] in ("build", "test", "check", "run"):
        return [UnresolvedEffect(reason="wrapper_command", fragment="cargo: build tool")]
    return []


@_REGISTRY.register("python", syntax="posix")
@_REGISTRY.register("python3", syntax="posix")
@_REGISTRY.register("node", syntax="posix")
def _interpreter(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """python/node [script] — 脚本解释器，无法静态分析。"""
    args = [a for a in argv[1:] if not _is_flag(a) and not a.startswith("-")]
    effects: list[FileEffect | UnresolvedEffect] = []
    if args:
        # 第一个非 flag 参数通常是脚本文件（读）
        effects.append(_effect_from_arg(args[0], "read"))
        effects.append(
            UnresolvedEffect(
                reason="eval_like",
                fragment=f"{argv[0]}: dynamic execution of {args[0]}",
            )
        )
    return effects


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


@_REGISTRY.register("invoke-webrequest", syntax="powershell")
@_REGISTRY.register("iwr", syntax="powershell")
@_REGISTRY.register("wget", syntax="powershell")
@_REGISTRY.register("curl", syntax="powershell")
def _ps_webrequest(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """Invoke-WebRequest / iwr / wget / curl — 下载文件。

    -OutFile file → write
    """
    for i, arg in enumerate(argv):
        if arg.lower() in ("-outfile", "-out_file"):
            if i + 1 < len(argv):
                return [_effect_from_arg(argv[i + 1], "write")]
    # 没有 -OutFile 时输出到管道，无文件效果
    return []


@_REGISTRY.register("invoke-restmethod", syntax="powershell")
@_REGISTRY.register("irm", syntax="powershell")
def _ps_restmethod(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """Invoke-RestMethod — REST API 调用（网络，无直接文件效果）。"""
    return []


@_REGISTRY.register("new-item", syntax="powershell")
@_REGISTRY.register("ni", syntax="powershell")
def _ps_new_item(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """New-Item -Path file -Type Directory/File → write。"""
    return _ps_path_param_effects(argv, "write")


@_REGISTRY.register("rename-item", syntax="powershell")
@_REGISTRY.register("ren", syntax="powershell")
@_REGISTRY.register("rni", syntax="powershell")
def _ps_rename_item(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """Rename-Item old new → 两者都是 write。"""
    paths = _ps_extract_paths(argv)
    return [_effect_from_arg(p, "write") for p in paths[:2]]


@_REGISTRY.register("start-process", syntax="powershell")
@_REGISTRY.register("saps", syntax="powershell")
def _ps_start_process(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """Start-Process — 启动进程。"""
    for i, arg in enumerate(argv):
        if arg.lower() in ("-filepath", "-file_path"):
            if i + 1 < len(argv):
                return [
                    UnresolvedEffect(
                        reason="wrapper_command",
                        fragment=f"Start-Process {argv[i+1]}: runs executable",
                    )
                ]
    return []


@_REGISTRY.register("measure-command", syntax="powershell")
def _ps_measure_command(argv: list[str]) -> list[FileEffect | UnresolvedEffect]:
    """Measure-Command — 性能测量，无文件效果。"""
    return []


# ── POSIX AST 分析器 ──


class PosixAnalyzer:
    """使用 tree-sitter-bash 解析 POSIX shell 命令。"""

    def __init__(self, registry: CommandSemanticsRegistry | None = None) -> None:
        self._registry = registry or _REGISTRY
        self._parser: Any = None
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
        node: Any,
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
        node: Any,
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
        node: Any,
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
        self._parser: Any = None
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
        node: Any,
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
        node: Any,
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
        node: Any,
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


# ── cmd.exe 分析器 ──


class CmdAnalyzer:
    """cmd.exe 命令的简单分词分析器。

    cmd.exe 没有 tree-sitter 语法文件，使用 shlex 分词 + 命令识别。
    设计为 fail-closed：不能识别的命令产生 unresolved 效果。
    """

    # 读文件命令：type, more, find, fc, comp, sort, dir
    _READ_COMMANDS = frozenset({
        "type", "more", "find", "fc", "comp", "sort", "dir",
    })
    # 写文件命令：copy, xcopy, mkdir, md, attrib, icacls, takeown
    _WRITE_COMMANDS = frozenset({
        "copy", "xcopy", "mkdir", "md", "attrib", "icacls", "takeown",
    })
    # 删除命令：del, erase, rmdir, rd
    _DELETE_COMMANDS = frozenset({
        "del", "erase", "rmdir", "rd",
    })
    # 移动/重命名：move, ren, rename
    _MOVE_COMMANDS = frozenset({
        "move", "ren", "rename",
    })
    # 无文件效果：echo, cls, ver, date, time, cd, chdir, set, prompt, title
    _SAFE_COMMANDS = frozenset({
        "echo", "cls", "ver", "date", "time", "cd", "chdir",
        "set", "prompt", "title", "color", "help",
    })

    def analyze(self, command: str) -> ShellAnalysis:
        import shlex

        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            tokens = command.split()

        if not tokens:
            return ShellAnalysis(
                resolved_paths=(),
                unresolved_effects=(),
                primary_command=None,
                shell_type="cmd",
                parse_error=False,
                ast_available=False,
            )

        resolved: list[Target] = []
        unresolved: list[UnresolvedEffect] = []
        seen: set[str] = set()

        # 检测重定向
        resolved_redirects = self._redirect_targets(command)
        for t in resolved_redirects:
            if t.value not in seen:
                seen.add(t.value)
                resolved.append(t)

        cmd_name = tokens[0].lower()
        args = [a.strip("\"'") for a in tokens[1:] if a not in (">", ">>", "<", "2>", "2>>", "1>", "1>>")]

        if cmd_name in self._SAFE_COMMANDS:
            return ShellAnalysis(
                resolved_paths=tuple(resolved),
                unresolved_effects=tuple(unresolved),
                primary_command=cmd_name,
                shell_type="cmd",
                parse_error=False,
                ast_available=False,
            )

        if cmd_name in self._READ_COMMANDS:
            for a in args:
                if a and not a.startswith("/") and a not in seen:
                    seen.add(a)
                    resolved.append(Target(
                        kind="path", value=a, access="read",
                        provenance="shell_literal",
                    ))

        elif cmd_name in self._WRITE_COMMANDS:
            if cmd_name in ("copy", "xcopy"):
                # copy src dst → src=read, dst=write
                if len(args) >= 2:
                    resolved.append(Target(
                        kind="path", value=args[0], access="read",
                        provenance="shell_literal",
                    ))
                    resolved.append(Target(
                        kind="path", value=args[1], access="write",
                        provenance="shell_literal",
                    ))
                elif len(args) == 1:
                    resolved.append(Target(
                        kind="path", value=args[0], access="read",
                        provenance="shell_literal",
                    ))
            else:
                for a in args:
                    if a and not a.startswith("/") and a not in seen:
                        seen.add(a)
                        resolved.append(Target(
                            kind="path", value=a, access="write",
                            provenance="shell_literal",
                        ))

        elif cmd_name in self._DELETE_COMMANDS:
            for a in args:
                if a and not a.startswith("/") and a not in seen:
                    seen.add(a)
                    resolved.append(Target(
                        kind="path", value=a, access="delete",
                        provenance="shell_literal",
                    ))

        elif cmd_name in self._MOVE_COMMANDS:
            # move/ren src dst → 两者都是 write
            for a in args[:2]:
                if a and not a.startswith("/") and a not in seen:
                    seen.add(a)
                    resolved.append(Target(
                        kind="path", value=a, access="write",
                        provenance="shell_literal",
                    ))

        else:
            # 未知命令 → unresolved
            unresolved.append(UnresolvedEffect(
                reason="wrapper_command",
                fragment=f"cmd.exe unknown command: {cmd_name}",
            ))

        return ShellAnalysis(
            resolved_paths=tuple(resolved),
            unresolved_effects=tuple(unresolved),
            primary_command=cmd_name,
            shell_type="cmd",
            parse_error=False,
            ast_available=False,
        )

    @staticmethod
    def _redirect_targets(command: str) -> list[Target]:
        """从 cmd 命令中提取重定向目标路径。

        识别 > file, >> file, < file, 2> file 等。
        """
        import shlex

        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return []

        results: list[Target] = []
        redirect_ops = {">", ">>", "1>", "1>>", "2>", "2>>", "<", "0<", "2<"}

        for i, tok in enumerate(tokens):
            if tok in redirect_ops and i + 1 < len(tokens):
                path = tokens[i + 1].strip("\"'")
                if path:
                    access: PermissionAccess = "write"
                    if "<" in tok:
                        access = "read"
                    results.append(Target(
                        kind="path", value=path, access=access,
                        provenance="shell_literal",
                    ))
        return results


def analyze_shell_command(
    command: str,
    shell_type: str = "posix",
) -> ShellAnalysis:
    """统一入口：按 shell 类型选择解析器。

    Args:
        command: shell 命令文本。
        shell_type: "posix"（bash/zsh/sh/fish）、"powershell" 或 "cmd"。

    Returns:
        ShellAnalysis 包含可确认路径和不可确认效果。
    """
    if shell_type == "powershell":
        analyzer = PowerShellAnalyzer()
    elif shell_type == "cmd":
        analyzer = CmdAnalyzer()
    else:
        analyzer = PosixAnalyzer()
    return analyzer.analyze(command)


# ── 辅助函数 ──


def _child_by_field(node: Any, field_name: str) -> Any | None:
    """获取 tree-sitter 节点的命名子节点（field child）。

    tree-sitter 0.25+ 的 Node 对象通常有 child_by_field_name 方法。
    """
    if hasattr(node, "child_by_field_name"):
        return node.child_by_field_name(field_name)
    return None


def _first_name_word(node: Any) -> Any | None:
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


def _is_sed_script(arg: str) -> bool:
    """判断参数是否为 sed 脚本（非文件路径）。

    典型模式：s/foo/bar/g, /pattern/d, 或是简单数字地址
    """
    # 替换命令模式
    if arg.startswith("s/") or arg.startswith("y/"):
        return True
    # 地址模式：/pattern/ 或数字
    if arg.startswith("/") and arg.endswith("/"):
        return True
    # 纯数字或 $（行地址）
    if arg in {"$"} or arg.isdigit():
        return True
    # 逗号分隔的范围地址：1,5 或 /pat1/,/pat2/
    if "," in arg:
        parts = arg.split(",")
        if all(p.strip().isdigit() or (p.strip().startswith("/") and p.strip().endswith("/")) or p.strip() == "$" for p in parts):
            return True
    return False


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
