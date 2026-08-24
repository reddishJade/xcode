from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class SandboxMode(StrEnum):
    """Sandbox 的文件系统访问级别。"""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class NetworkAccess(StrEnum):
    """Sandbox 内的网络访问级别。"""

    DENY = "deny"
    ALLOW = "allow"


@dataclass(frozen=True)
class SandboxPolicy:
    """一次 Agent 进程执行所使用的隔离策略。"""

    project_root: Path
    mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    network_access: NetworkAccess = NetworkAccess.DENY
    writable_roots: tuple[Path, ...] = ()
    protected_workspace_paths: tuple[str, ...] = (
        ".git",
        ".agents",
        ".xcode",
    )


@dataclass(frozen=True)
class SandboxedCommand:
    """经过平台 sandbox 包装后可交给进程启动器的命令。"""

    argv: tuple[str, ...]
    cwd: Path


class CommandSandbox(Protocol):
    """把原始命令转换为平台 sandbox 启动命令。"""

    def wrap(self, argv: list[str], cwd: Path) -> SandboxedCommand: ...
