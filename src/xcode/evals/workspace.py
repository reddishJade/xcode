"""从精确 Git 版本创建不含控制面材料的隔离 Trial 工作区。"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from .schema import Task


class WorkspaceError(RuntimeError):
    """工作区无法被确定性恢复。"""


@dataclass(frozen=True)
class TrialWorkspace:
    """一次 Trial 独占的工作区及初始内容指纹。"""

    root: Path
    revision: str
    initial_digest: str
    initial_files: dict[str, bytes]


class GitWorkspaceFactory:
    """通过 `git archive` 恢复提交，不向 Agent 暴露仓库历史。"""

    def __init__(self, *, repository: Path, workspace_root: Path) -> None:
        self._repository = repository.resolve()
        self._workspace_root = workspace_root.resolve()

    def create(self, task: Task, trial_id: str) -> TrialWorkspace:
        """为 Trial 创建全新目录，并校验来源版本确实存在。"""
        destination = self._workspace_root / trial_id
        if destination.exists():
            raise WorkspaceError(f"trial workspace already exists: {destination}")
        destination.mkdir(parents=True)

        revision = self._resolve_revision(task.source.revision)
        try:
            self._extract_revision(revision, destination)
        except Exception:
            shutil.rmtree(destination)
            raise
        initial_files = workspace_files(destination)
        return TrialWorkspace(
            root=destination,
            revision=revision,
            initial_digest=workspace_digest(destination),
            initial_files=initial_files,
        )

    def _resolve_revision(self, revision: str) -> str:
        completed = subprocess.run(
            ("git", "rev-parse", "--verify", f"{revision}^{{commit}}"),
            cwd=self._repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise WorkspaceError(completed.stderr.strip() or "invalid Git revision")
        return completed.stdout.strip()

    def _extract_revision(self, revision: str, destination: Path) -> None:
        with TemporaryDirectory(dir=self._workspace_root) as temporary:
            archive = Path(temporary) / "workspace.tar"
            completed = subprocess.run(
                (
                    "git",
                    "archive",
                    "--format=tar",
                    f"--output={archive}",
                    revision,
                ),
                cwd=self._repository,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise WorkspaceError(completed.stderr.strip() or "git archive failed")
            with tarfile.open(archive, mode="r:") as bundle:
                bundle.extractall(destination, filter="data")


def workspace_digest(root: Path) -> str:
    """计算稳定内容指纹，忽略目录元数据和文件时间。"""
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def workspace_files(root: Path) -> dict[str, bytes]:
    """读取工作区普通文件，用于结束后生成 patch 和 policy 证据。"""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def changed_paths(initial_files: dict[str, bytes], root: Path) -> tuple[str, ...]:
    """返回新增、删除或内容变化的稳定路径集合。"""
    current = workspace_files(root)
    return tuple(
        sorted(
            path
            for path in set(initial_files) | set(current)
            if initial_files.get(path) != current.get(path)
        )
    )


def workspace_patch(initial_files: dict[str, bytes], root: Path) -> str:
    """生成可审计 unified patch；二进制变化用明确标记保存。"""
    current = workspace_files(root)
    parts: list[str] = []
    for relative in changed_paths(initial_files, root):
        before = initial_files.get(relative)
        after = current.get(relative)
        try:
            before_lines = before.decode().splitlines(keepends=True) if before else []
            after_lines = after.decode().splitlines(keepends=True) if after else []
        except UnicodeDecodeError:
            parts.append(f"Binary files a/{relative} and b/{relative} differ\n")
            continue
        parts.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(parts)
