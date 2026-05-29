from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass


SHELL_NAMES = frozenset(
    {
        "auto",
        "pwsh",
        "powershell",
        "cmd",
        "bash",
        "zsh",
        "sh",
        "fish",
    }
)


@dataclass(frozen=True)
class ShellSpec:
    name: str
    command_prefix: tuple[str, ...]
    syntax: str  # "powershell" | "cmd" | "posix" — shell command grammar only


_KNOWN_SHELLS: dict[str, ShellSpec] = {
    "pwsh": ShellSpec(
        name="pwsh",
        command_prefix=("pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"),
        syntax="powershell",
    ),
    "powershell": ShellSpec(
        name="powershell",
        command_prefix=(
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
        ),
        syntax="powershell",
    ),
    "cmd": ShellSpec(
        name="cmd",
        command_prefix=("cmd", "/d", "/c"),
        syntax="cmd",
    ),
    "bash": ShellSpec(
        name="bash",
        command_prefix=("bash", "--noprofile", "--norc", "-c"),
        syntax="posix",
    ),
    "zsh": ShellSpec(
        name="zsh",
        command_prefix=("zsh", "-f", "-c"),
        syntax="posix",
    ),
    "sh": ShellSpec(
        name="sh",
        command_prefix=("sh", "-c"),
        syntax="posix",
    ),
    "fish": ShellSpec(
        name="fish",
        command_prefix=("fish", "-c"),
        syntax="posix",
    ),
}


def detect_shell(config: str = "auto") -> ShellSpec:
    """检测并返回当前环境可用的 ShellSpec。

    参数:
        config: "auto" 自动检测，或显式指定 shell 名称。

    返回:
        对应的 ShellSpec。

    异常:
        ValueError: 未知的 shell 名称。
        RuntimeError: 显式指定的 shell 不在 PATH 上。
    """
    if config != "auto":
        return _resolve_explicit_shell(config)
    if sys.platform == "win32":
        return _detect_windows_shell()
    return _detect_posix_shell()


def build_shell_argv(spec: ShellSpec, command: str) -> list[str]:
    return [*spec.command_prefix, command]


def _resolve_explicit_shell(config: str) -> ShellSpec:
    if config not in SHELL_NAMES:
        raise ValueError(
            f"unknown shell: {config!r}. "
            f"Valid options: {', '.join(sorted(SHELL_NAMES))}"
        )
    spec = _KNOWN_SHELLS[config]
    if not _is_on_path(spec):
        raise RuntimeError(
            f"configured shell {config!r} not found on PATH. "
            "Install it or switch the 'shell' config back to 'auto'."
        )
    return spec


def _is_on_path(spec: ShellSpec) -> bool:
    return shutil.which(spec.command_prefix[0]) is not None


def _detect_windows_shell() -> ShellSpec:
    for name in ("pwsh", "powershell", "bash", "cmd"):
        spec = _KNOWN_SHELLS[name]
        if shutil.which(spec.command_prefix[0]):
            return spec
    return _KNOWN_SHELLS["cmd"]


def _detect_posix_shell() -> ShellSpec:
    shell_env = os.environ.get("SHELL")
    if shell_env:
        name = os.path.basename(shell_env)
        if name in _KNOWN_SHELLS:
            spec = _KNOWN_SHELLS[name]
            if shutil.which(spec.command_prefix[0]):
                return spec
    for name in ("bash", "zsh", "sh"):
        spec = _KNOWN_SHELLS[name]
        if shutil.which(spec.command_prefix[0]):
            return spec
    return _KNOWN_SHELLS["sh"]
