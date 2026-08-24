from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
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

        placeholders = _prepare_protected_placeholders(self._policy)
        try:
            args = [str(self._bwrap_path), "--new-session", "--die-with-parent"]
            args.extend(self._filesystem_args())
            args.extend(
                (
                    "--unshare-user",
                    "--disable-userns",
                    "--unshare-pid",
                    "--unshare-ipc",
                    "--unshare-uts",
                )
            )
            if self._policy.network_access is NetworkAccess.DENY:
                args.append("--unshare-net")
            args.extend(("--proc", "/proc"))
            args.extend(("--chdir", str(command_cwd)))
            args.extend(("--cap-drop", "ALL", "--"))
            args.extend(argv)
        except (OSError, ValueError):
            _protected_placeholder_finalizer(placeholders)()
            raise
        return SandboxedCommand(
            argv=tuple(args),
            cwd=command_cwd,
            finalize=(
                _protected_placeholder_finalizer(placeholders) if placeholders else None
            ),
        )

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
        for unreadable in policy.unreadable_roots:
            if unreadable.is_dir():
                args.extend(("--tmpfs", str(unreadable)))
            else:
                args.extend(("--ro-bind", "/dev/null", str(unreadable)))
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

    unreadable_roots: list[Path] = []
    for raw_root in policy.unreadable_roots:
        root = raw_root.resolve(strict=True)
        unreadable_roots.append(root)

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
        unreadable_roots=tuple(_minimal_roots(unreadable_roots)),
        protected_workspace_paths=policy.protected_workspace_paths,
    )


def _existing_protected_paths(policy: SandboxPolicy) -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative in policy.protected_workspace_paths:
        path = policy.project_root / relative
        if path.is_symlink():
            raise ValueError(f"protected workspace path must not be a symlink: {path}")
        if path.exists():
            paths.append(path)
    return tuple(paths)


@dataclass(frozen=True)
class _ProtectedPlaceholder:
    path: Path
    device: int
    inode: int
    descriptor: int


def _prepare_protected_placeholders(
    policy: SandboxPolicy,
) -> tuple[_ProtectedPlaceholder, ...]:
    if policy.mode is not SandboxMode.WORKSPACE_WRITE:
        return ()
    placeholders: list[_ProtectedPlaceholder] = []
    try:
        for relative in policy.protected_workspace_paths:
            path = policy.project_root / relative
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                if path.is_symlink():
                    raise ValueError(
                        f"protected workspace path must not be a symlink: {path}"
                    ) from None
                continue
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
            except OSError:
                path.rmdir()
                raise
            metadata = os.fstat(descriptor)
            placeholders.append(
                _ProtectedPlaceholder(
                    path=path,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    descriptor=descriptor,
                )
            )
    except (OSError, ValueError):
        _protected_placeholder_finalizer(tuple(placeholders))()
        raise
    return tuple(placeholders)


def _protected_placeholder_finalizer(
    placeholders: tuple[_ProtectedPlaceholder, ...],
) -> Callable[[], str | None]:
    completed = False

    def finalize() -> str | None:
        nonlocal completed
        if completed:
            return None
        completed = True
        violations: list[str] = []
        for placeholder in reversed(placeholders):
            path = placeholder.path
            try:
                try:
                    metadata = path.lstat()
                except FileNotFoundError:
                    continue
                if (
                    metadata.st_dev != placeholder.device
                    or metadata.st_ino != placeholder.inode
                ):
                    violations.append(f"{path} (placeholder identity changed)")
                    continue
                try:
                    path.rmdir()
                except OSError as exc:
                    violations.append(f"{path} (cleanup failed: {exc})")
            finally:
                os.close(placeholder.descriptor)
        if not violations:
            return None
        return "sandbox failed to clean protected path placeholder: " + ", ".join(
            violations
        )

    return finalize


def _deduplicate_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []
    for root in sorted(set(roots), key=lambda path: (len(path.parts), str(path))):
        if not any(root == existing for existing in unique):
            unique.append(root)
    return tuple(unique)


def _minimal_roots(roots: list[Path]) -> tuple[Path, ...]:
    minimal: list[Path] = []
    for root in sorted(set(roots), key=lambda path: (len(path.parts), str(path))):
        if any(_is_relative_to(root, parent) for parent in minimal):
            continue
        minimal.append(root)
    return tuple(minimal)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
