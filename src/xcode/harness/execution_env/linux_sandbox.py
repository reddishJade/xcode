from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .sandbox import (
    CommandSandbox,
    NetworkAccess,
    SandboxedCommand,
    SandboxMode,
    SandboxPolicy,
)


class SandboxUnavailableError(RuntimeError):
    """当前主机无法提供请求的 OS sandbox。"""


class LinuxBubblewrapSandbox(CommandSandbox):
    """使用 bubblewrap 为 Agent 子进程构造 Linux OS sandbox。"""

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        bwrap_path: Path | None = None,
    ) -> None:
        if sys.platform != "linux":
            raise SandboxUnavailableError("Linux bubblewrap sandbox requires Linux")
        self._policy = _normalize_policy(policy)
        self._bwrap_path = _resolve_bwrap_path(bwrap_path)

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    def wrap(self, argv: list[str], cwd: Path) -> SandboxedCommand:
        if not argv:
            raise ValueError("sandbox command argv must not be empty")

        command_cwd = cwd.resolve(strict=True)
        if not _is_relative_to(command_cwd, self._policy.project_root):
            raise ValueError("sandbox command cwd must stay inside the project root")

        args = [str(self._bwrap_path), "--new-session", "--die-with-parent"]
        args.extend(self._filesystem_args())
        args.extend(("--unshare-user", "--unshare-pid", "--unshare-ipc"))
        if self._policy.network_access is NetworkAccess.DENY:
            args.append("--unshare-net")
        args.extend(("--proc", "/proc"))
        args.extend(("--chdir", str(command_cwd)))
        args.extend(("--cap-drop", "ALL", "--"))
        args.extend(argv)
        return SandboxedCommand(argv=tuple(args), cwd=command_cwd)

    def _filesystem_args(self) -> list[str]:
        policy = self._policy
        if policy.mode is SandboxMode.DANGER_FULL_ACCESS:
            return ["--bind", "/", "/", "--dev", "/dev"]

        args = ["--ro-bind", "/", "/", "--dev", "/dev"]
        if policy.mode is SandboxMode.WORKSPACE_WRITE:
            writable_roots = _deduplicate_roots(
                (policy.project_root, *policy.writable_roots)
            )
            for root in writable_roots:
                args.extend(("--bind", str(root), str(root)))
            for protected in _existing_protected_paths(policy):
                args.extend(("--ro-bind", str(protected), str(protected)))
        return args


def _resolve_bwrap_path(explicit: Path | None) -> Path:
    raw_path = str(explicit) if explicit is not None else shutil.which("bwrap")
    if raw_path is None:
        raise SandboxUnavailableError(
            "bubblewrap executable not found; install 'bwrap' or select "
            "danger-full-access"
        )
    path = Path(raw_path).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SandboxUnavailableError(f"bubblewrap is not executable: {path}")
    return path


def _normalize_policy(policy: SandboxPolicy) -> SandboxPolicy:
    project_root = policy.project_root.resolve(strict=True)
    if not project_root.is_dir():
        raise ValueError("sandbox project root must be a directory")

    writable_roots: list[Path] = []
    for raw_root in policy.writable_roots:
        root = raw_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"sandbox writable root must be a directory: {root}")
        writable_roots.append(root)

    for relative in policy.protected_workspace_paths:
        protected = Path(relative)
        if protected.is_absolute() or ".." in protected.parts:
            raise ValueError(
                f"protected workspace path must be project-relative: {relative}"
            )

    return SandboxPolicy(
        project_root=project_root,
        mode=policy.mode,
        network_access=policy.network_access,
        writable_roots=tuple(writable_roots),
        protected_workspace_paths=policy.protected_workspace_paths,
    )


def _existing_protected_paths(policy: SandboxPolicy) -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative in policy.protected_workspace_paths:
        path = (policy.project_root / relative).resolve()
        if path.exists() and _is_relative_to(path, policy.project_root):
            paths.append(path)
    return tuple(paths)


def _deduplicate_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []
    for root in sorted(set(roots), key=lambda path: (len(path.parts), str(path))):
        if not any(root == existing for existing in unique):
            unique.append(root)
    return tuple(unique)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
