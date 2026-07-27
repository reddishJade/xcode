"""保守的 Shell 命令分类。

这里不模拟命令的完整文件副作用。分类器只识别少量确定的只读命令；
动态语法、未知命令和写操作由执行模式决定，明确危险的宿主操作直接拒绝。
真正的文件和网络边界必须由 OS sandbox 提供。
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .permission_model import Action, Constraint, Target, UnresolvedEffect

ShellType = Literal["posix", "powershell", "cmd"]
ShellUnresolvedPolicy = Literal["allow", "ask", "deny"]

_POSIX_READ_COMMANDS = frozenset(
    {
        "ack",
        "cat",
        "dir",
        "grep",
        "head",
        "less",
        "ls",
        "more",
        "realpath",
        "rg",
        "tail",
    }
)
_POSIX_NO_EFFECT_COMMANDS = frozenset({"false", "pwd", "true", "uname", "whoami"})
_POSIX_MUTATING_COMMANDS = frozenset({"cp", "mkdir", "mv", "rm", "touch"})
_POWERSHELL_READ_COMMANDS = frozenset({"get-childitem", "get-content", "select-string"})
_POWERSHELL_NO_EFFECT_COMMANDS = frozenset({"get-date", "get-location"})
_CMD_READ_COMMANDS = frozenset({"dir", "more", "type"})
_CMD_NO_EFFECT_COMMANDS = frozenset({"cls", "echo", "ver"})

_SEPARATORS = frozenset({";", "&&", "||", "|"})
_UNSAFE_CONTROL = frozenset({"&", "(", ")"})
_REDIRECTIONS = frozenset({"<", ">", "<<", ">>", "<<<"})
_FIND_EXECUTORS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})
_GLOB_CHARS = frozenset("*?[")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)


@dataclass(frozen=True)
class ShellAnalysis:
    """权限层消费的最小 Shell 分类结果。"""

    resolved_paths: tuple[Target, ...]
    unresolved_effects: tuple[UnresolvedEffect, ...]
    primary_command: str | None
    shell_type: ShellType
    parse_error: bool
    classification_available: bool


class ShellAnalysisPolicyEvaluator:
    """把保守分类结果转换为权限约束。"""

    def evaluate(
        self,
        action: Action,
        unresolved_policy: ShellUnresolvedPolicy = "ask",
    ) -> tuple[Constraint, ...]:
        constraints: list[Constraint] = []
        for effect in action.unresolved_effects:
            dangerous = effect.reason == "dangerous_command"
            if not dangerous and unresolved_policy == "allow":
                continue
            constraints.append(
                Constraint(
                    decision="deny" if dangerous else unresolved_policy,
                    source="shell_policy",
                    reason=f"{effect.reason}: {effect.fragment}",
                )
            )
        return tuple(constraints)


class PosixAnalyzer:
    def analyze(self, command: str) -> ShellAnalysis:
        return _analyze_posix(command)


class PowerShellAnalyzer:
    def analyze(self, command: str) -> ShellAnalysis:
        return _analyze_simple_shell(command, "powershell")


class CmdAnalyzer:
    def analyze(self, command: str) -> ShellAnalysis:
        return _analyze_simple_shell(command, "cmd")


def analyze_shell_command(
    command: str,
    shell_type: str = "posix",
) -> ShellAnalysis:
    """按 Shell 类型执行保守分类。"""
    if shell_type == "powershell":
        return PowerShellAnalyzer().analyze(command)
    if shell_type == "cmd":
        return CmdAnalyzer().analyze(command)
    return PosixAnalyzer().analyze(command)


def _analyze_posix(command: str) -> ShellAnalysis:
    try:
        tokens = _posix_tokens(command)
    except ValueError:
        return _unresolved(command, "posix", "parse_error", "invalid quoting", True)
    if not tokens:
        return _empty("posix")

    primary = _primary_command(tokens)
    dangerous = _dangerous_posix(tokens)
    if dangerous is not None:
        return _result(
            "posix",
            primary,
            unresolved=(
                UnresolvedEffect(
                    reason="dangerous_command",
                    fragment=dangerous,
                ),
            ),
        )

    dynamic = _dynamic_effect(command, tokens)
    if dynamic is not None:
        return _result("posix", primary, unresolved=(dynamic,))

    segments, unsupported = _segments(tokens)
    if unsupported is not None:
        return _result(
            "posix",
            primary,
            unresolved=(
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment=unsupported,
                ),
            ),
        )

    paths: list[Target] = []
    unresolved: list[UnresolvedEffect] = []
    for segment in segments:
        name, args = _command_and_args(segment)
        if name is None:
            continue
        if name == "find":
            if any(arg.lower() in _FIND_EXECUTORS for arg in args):
                unresolved.append(
                    UnresolvedEffect(
                        reason="wrapper_command",
                        fragment="find executes or deletes paths",
                    )
                )
                continue
            paths.extend(_find_paths(args))
            continue
        if name in _POSIX_NO_EFFECT_COMMANDS:
            continue
        if name in _POSIX_READ_COMMANDS:
            paths.extend(_read_paths(name, args))
            continue
        if name in _POSIX_MUTATING_COMMANDS:
            paths.extend(_mutating_paths(name, args))
            unresolved.append(
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment=f"write command requires approval: {name}",
                )
            )
            continue
        unresolved.append(
            UnresolvedEffect(
                reason="wrapper_command",
                fragment=f"command requires approval: {name}",
            )
        )

    return _result(
        "posix",
        primary,
        paths=_deduplicate_paths(paths),
        unresolved=tuple(unresolved),
    )


def _analyze_simple_shell(
    command: str, shell_type: Literal["powershell", "cmd"]
) -> ShellAnalysis:
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return _unresolved(
            command,
            shell_type,
            "parse_error",
            "invalid quoting",
            True,
        )
    if not tokens:
        return _empty(shell_type)

    primary = _basename(tokens[0])
    lowered = command.lower()
    if _dangerous_simple(primary, tokens[1:]):
        return _result(
            shell_type,
            primary,
            unresolved=(
                UnresolvedEffect(
                    reason="dangerous_command",
                    fragment=command,
                ),
            ),
        )
    if any(marker in command for marker in ("$", "`", "|", ";", ">", "<", "&")):
        return _result(
            shell_type,
            primary,
            unresolved=(
                UnresolvedEffect(
                    reason="wrapper_command",
                    fragment="dynamic or compound shell syntax",
                ),
            ),
        )

    read_commands = (
        _POWERSHELL_READ_COMMANDS if shell_type == "powershell" else _CMD_READ_COMMANDS
    )
    no_effect_commands = (
        _POWERSHELL_NO_EFFECT_COMMANDS
        if shell_type == "powershell"
        else _CMD_NO_EFFECT_COMMANDS
    )
    if primary in no_effect_commands:
        return _result(shell_type, primary)
    if primary in read_commands:
        paths = [
            _path_target(token.strip("\"'"))
            for token in tokens[1:]
            if token and not token.startswith(("-", "/"))
        ]
        return _result(
            shell_type,
            primary,
            paths=_deduplicate_paths(paths),
        )
    return _result(
        shell_type,
        primary,
        unresolved=(
            UnresolvedEffect(
                reason="wrapper_command",
                fragment=f"command requires approval: {lowered}",
            ),
        ),
    )


def _posix_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _segments(tokens: list[str]) -> tuple[list[list[str]], str | None]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _SEPARATORS:
            if segments[-1]:
                segments.append([])
            continue
        if token in _UNSAFE_CONTROL:
            return (segments, f"unsupported shell control operator: {token}")
        if token in _REDIRECTIONS or any(char in token for char in "<>"):
            return (segments, "shell redirection requires approval")
        segments[-1].append(token)
    return ([segment for segment in segments if segment], None)


def _primary_command(tokens: list[str]) -> str | None:
    for token in tokens:
        if token in _SEPARATORS | _UNSAFE_CONTROL | _REDIRECTIONS:
            continue
        if _ASSIGNMENT.fullmatch(token):
            continue
        return _basename(token)
    return None


def _command_and_args(segment: list[str]) -> tuple[str | None, list[str]]:
    index = 0
    while index < len(segment) and _ASSIGNMENT.fullmatch(segment[index]):
        index += 1
    if index >= len(segment):
        return (None, [])
    return (_basename(segment[index]), segment[index + 1 :])


def _dynamic_effect(
    command: str,
    tokens: list[str],
) -> UnresolvedEffect | None:
    if "$(" in command or "`" in command:
        return UnresolvedEffect(
            reason="command_substitution",
            fragment="command substitution",
        )
    if "$" in command:
        return UnresolvedEffect(reason="variable_expansion", fragment="shell variable")
    if any(any(char in token for char in _GLOB_CHARS) for token in tokens):
        return UnresolvedEffect(reason="glob", fragment="shell glob")
    return None


def _dangerous_posix(tokens: list[str]) -> str | None:
    segments, _ = _segments(tokens)
    for segment in segments:
        name, args = _command_and_args(segment)
        lowered = [arg.lower() for arg in args]
        if name in {"mkfs", "poweroff", "reboot", "shutdown"}:
            return "host-level destructive command"
        if name in {"sudo", "doas", "su"}:
            return "privilege escalation command"
        if name == "git" and "reset" in lowered and "--hard" in lowered:
            return "git reset --hard discards working tree changes"
        if (
            name == "git"
            and "clean" in lowered
            and any("f" in arg.lstrip("-") for arg in lowered if arg.startswith("-"))
        ):
            return "git clean -f deletes untracked files"
        if name == "rm" and _is_root_recursive_delete(lowered):
            return "recursive deletion of the filesystem root"
    return None


def _dangerous_simple(primary: str, args: list[str]) -> bool:
    lowered = {arg.lower() for arg in args}
    if primary in {"format", "shutdown"}:
        return True
    if primary == "remove-item" and "-recurse" in lowered and "-force" in lowered:
        return True
    return False


def _is_root_recursive_delete(args: list[str]) -> bool:
    recursive = any(
        arg in {"--recursive", "--force"}
        or (arg.startswith("-") and "r" in arg and "f" in arg)
        for arg in args
    )
    targets = {arg for arg in args if not arg.startswith("-") and arg != "--"}
    return recursive and bool(targets & {"/", "/*"})


def _read_paths(command: str, args: list[str]) -> list[Target]:
    positional: list[str] = []
    skip_next = False
    options_with_values = {"-c", "-n"} if command in {"head", "tail"} else set()
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in options_with_values:
            skip_next = True
            continue
        if arg and not arg.startswith("-"):
            positional.append(arg)
    if command in {"grep", "rg", "ack"} and positional:
        positional = positional[1:]
    return [_path_target(arg) for arg in positional]


def _mutating_paths(command: str, args: list[str]) -> list[Target]:
    positional = [arg for arg in args if arg and not arg.startswith("-")]
    if command == "cp" and len(positional) >= 2:
        return [
            _path_target(positional[-2]),
            _path_target(positional[-1], access="write"),
        ]
    access: Literal["write", "delete"] = "delete" if command == "rm" else "write"
    return [_path_target(arg, access=access) for arg in positional]


def _find_paths(args: list[str]) -> list[Target]:
    paths: list[Target] = []
    for arg in args:
        if arg.startswith(("-", "!", "(")):
            break
        paths.append(_path_target(arg))
    return paths


def _path_target(
    path: str,
    *,
    access: Literal["read", "write", "delete"] = "read",
) -> Target:
    return Target(
        kind="path",
        value=path,
        access=access,
        provenance="shell_literal",
    )


def _deduplicate_paths(paths: list[Target]) -> tuple[Target, ...]:
    seen: set[str] = set()
    result: list[Target] = []
    for target in paths:
        if target.value in seen:
            continue
        seen.add(target.value)
        result.append(target)
    return tuple(result)


def _basename(command: str) -> str:
    return Path(command.strip("\"'")).name.lower()


def _empty(shell_type: ShellType) -> ShellAnalysis:
    return _result(shell_type, None)


def _unresolved(
    command: str,
    shell_type: ShellType,
    reason: Literal["parse_error", "unsupported_shell"],
    fragment: str,
    parse_error: bool,
) -> ShellAnalysis:
    return ShellAnalysis(
        resolved_paths=(),
        unresolved_effects=(
            UnresolvedEffect(reason=reason, fragment=f"{fragment}: {command}"),
        ),
        primary_command=None,
        shell_type=shell_type,
        parse_error=parse_error,
        classification_available=True,
    )


def _result(
    shell_type: ShellType,
    primary: str | None,
    *,
    paths: tuple[Target, ...] = (),
    unresolved: tuple[UnresolvedEffect, ...] = (),
) -> ShellAnalysis:
    return ShellAnalysis(
        resolved_paths=paths,
        unresolved_effects=unresolved,
        primary_command=primary,
        shell_type=shell_type,
        parse_error=False,
        classification_available=True,
    )
