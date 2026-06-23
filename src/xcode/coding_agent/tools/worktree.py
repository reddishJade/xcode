"""托管实现任务使用的 Git worktree 隔离工具。

worktree 元数据持久化到 ``.local/worktrees/index.json``，进程重启后可恢复。
``prune_stale`` 清理 git 已 prune 但目录残留的孤儿 worktree。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
import subprocess

import filelock

from xcode.harness.skills import ToolInput, ToolSpec

logger = logging.getLogger("xcode.coding_agent.tools.worktree")

CommandRunner = Callable[[list[str], Path], str]


@dataclass(frozen=True)
class WorktreeTask:
    id: str
    path: Path
    branch: str


@dataclass(frozen=True)
class WorktreeInfo:
    """list() 返回的单个 worktree 视图。"""

    id: str
    path: Path
    branch: str
    dirty: bool
    exists: bool


class WorktreeTaskRunner:
    def __init__(
        self,
        repo_root: Path,
        worktrees_dir: Path | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.worktrees_dir = worktrees_dir or self.repo_root / ".local" / "worktrees"
        self.command_runner = command_runner or _run_command
        self.index_path = self.worktrees_dir / "index.json"
        self._lock_path = self.worktrees_dir / ".index.lock"
        self.tasks: dict[str, WorktreeTask] = {}
        self._load_index()
        self._reconcile_with_git()

    def _load_index(self) -> None:
        """从 index.json 恢复已知 worktree 列表。"""
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("failed to load worktree index, starting empty")
            return
        for entry in data.get("worktrees", []):
            try:
                task = WorktreeTask(
                    id=str(entry["id"]),
                    path=Path(entry["path"]),
                    branch=str(entry["branch"]),
                )
                self.tasks[task.id] = task
            except (KeyError, TypeError):
                continue

    def _save_index(self) -> None:
        """原子写 index.json。调用方必须持锁。"""
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "worktrees": [
                {"id": t.id, "path": str(t.path), "branch": t.branch}
                for t in self.tasks.values()
            ]
        }
        tmp_path = self.index_path.with_name(
            f".{self.index_path.name}.{uuid.uuid4().hex}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, self.index_path)

    def _index_lock(self) -> filelock.FileLock:
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        return filelock.FileLock(self._lock_path, timeout=10.0)

    def _reconcile_with_git(self) -> None:
        """对照 git worktree list 校验 index，清理失效条目并补录 git 已知但 index 缺失的 xcode worktree。"""
        try:
            output = self.command_runner(
                ["git", "worktree", "list", "--porcelain"], self.repo_root
            )
        except Exception:
            logger.debug("git worktree list failed during reconcile", exc_info=True)
            return
        git_paths: dict[Path, str] = {}
        current_worktree: dict[str, str] = {}
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("worktree "):
                wt_path = Path(line[len("worktree ") :]).resolve()
                current_worktree = {"path": str(wt_path)}
                git_paths[wt_path] = ""
            elif line.startswith("branch "):
                branch = line[len("branch ") :]
                if current_worktree:
                    # 记录 branch 到最后一条 worktree
                    # git_paths 的 value 用空串占位，这里用单独 dict 跟踪
                    git_paths[next(reversed(git_paths))] = branch
        # 清理 index 中 git 不再注册的条目
        removed_any = False
        for task_id in list(self.tasks.keys()):
            task = self.tasks[task_id]
            if task.path.resolve() not in git_paths:
                logger.info(
                    "reconcile: drop stale worktree %s (path %s not in git)",
                    task_id,
                    task.path,
                )
                del self.tasks[task_id]
                removed_any = True
        # 补录 git 已知、branch 以 xcode/ 为前缀但不在 index 的 worktree
        known_paths = {t.path.resolve() for t in self.tasks.values()}
        for wt_path, branch in git_paths.items():
            if wt_path in known_paths:
                continue
            if not branch.startswith("xcode/"):
                continue
            if not wt_path.is_relative_to(self.worktrees_dir.resolve()):
                continue
            # 从 path 推断 id（worktrees_dir/<id>）
            try:
                task_id = wt_path.relative_to(self.worktrees_dir.resolve()).name
            except ValueError:
                continue
            if task_id in self.tasks:
                continue
            self.tasks[task_id] = WorktreeTask(id=task_id, path=wt_path, branch=branch)
            removed_any = True  # 标记需要保存
            logger.info("reconcile: recover unindexed worktree %s", task_id)
        if removed_any:
            with self._index_lock():
                self._save_index()

    def create(self, name: str) -> WorktreeTask:
        task_id = uuid.uuid4().hex[:8]
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)[
            :32
        ]
        branch = f"xcode/{safe_name or 'task'}-{task_id}"
        path = self.worktrees_dir / task_id
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self.command_runner(
            ["git", "worktree", "add", "-b", branch, str(path), "HEAD"], self.repo_root
        )
        task = WorktreeTask(task_id, path, branch)
        with self._index_lock():
            self.tasks[task_id] = task
            self._save_index()
        return task

    def remove(self, task_id: str, force: bool = False, prune: bool = False) -> str:
        task = self.tasks.get(task_id)
        if task is None:
            return f"unknown worktree task: {task_id}"

        if not force:
            dirty_check = self._check_dirty(task)
            if dirty_check:
                return dirty_check
            unpushed_check = self._check_unpushed_commits(task)
            if unpushed_check:
                return unpushed_check

        cmd = ["git", "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(str(task.path))

        self.command_runner(cmd, self.repo_root)
        if prune:
            try:
                self.command_runner(
                    ["git", "worktree", "prune", "--expire=now"], self.repo_root
                )
            except Exception:
                logger.debug("git worktree prune failed", exc_info=True)
        with self._index_lock():
            del self.tasks[task_id]
            self._save_index()
        return f"removed worktree task {task_id}"

    def list(self) -> list[WorktreeInfo]:
        """列出所有已知 worktree 及其 dirty/exists 状态。"""
        try:
            output = self.command_runner(
                ["git", "worktree", "list", "--porcelain"], self.repo_root
            )
            git_paths = {
                Path(line[len("worktree ") :].strip()).resolve()
                for line in output.splitlines()
                if line.startswith("worktree ")
            }
        except Exception:
            logger.debug("git worktree list failed in list()", exc_info=True)
            git_paths = set()

        infos: list[WorktreeInfo] = []
        for task in self.tasks.values():
            exists = task.path.resolve() in git_paths
            dirty = False
            if exists:
                dirty = self._is_dirty(task)
            infos.append(
                WorktreeInfo(
                    id=task.id,
                    path=task.path,
                    branch=task.branch,
                    dirty=dirty,
                    exists=exists,
                )
            )
        return sorted(infos, key=lambda i: i.id)

    def prune_stale(self) -> list[Path]:
        """清理 worktrees_dir 下不在 index 中的孤儿 worktree 目录。

        仅清理经 ``git rev-parse --is-inside-worktree`` 确认为 worktree 的目录，
        非工作树的目录保留并记录 warning。返回被清理的 path 列表。
        """
        if not self.worktrees_dir.exists():
            return []
        cleaned: list[Path] = []
        known_paths = {t.path.resolve() for t in self.tasks.values()}
        worktrees_root = self.worktrees_dir.resolve()
        for child in self.worktrees_dir.iterdir():
            if child.name.startswith(".") or child.name == "index.json":
                continue
            if not child.is_dir():
                continue
            child_resolved = child.resolve()
            if child_resolved in known_paths:
                continue
            # 必须在 worktrees_dir 内（防止 symlink 逃逸）
            try:
                child_resolved.relative_to(worktrees_root)
            except ValueError:
                continue
            if not self._is_git_worktree(child):
                logger.warning("prune_stale: skip non-worktree dir %s", child)
                continue
            try:
                self.command_runner(
                    ["git", "worktree", "remove", "--force", str(child)],
                    self.repo_root,
                )
            except Exception:
                # git worktree remove 失败则直接删目录
                try:
                    shutil.rmtree(child)
                except OSError:
                    logger.warning("prune_stale: failed to remove %s", child)
                    continue
            cleaned.append(child)
        # 清理 git 内部元数据
        try:
            self.command_runner(
                ["git", "worktree", "prune", "--expire=now"], self.repo_root
            )
        except Exception:
            logger.debug("git worktree prune failed in prune_stale", exc_info=True)
        return cleaned

    def _is_git_worktree(self, path: Path) -> bool:
        try:
            output = self.command_runner(
                ["git", "rev-parse", "--is-inside-worktree"], path
            )
            return output.strip() == "true"
        except Exception:
            return False

    def _is_dirty(self, task: WorktreeTask) -> bool:
        try:
            status_out = self.command_runner(
                ["git", "status", "--porcelain"], task.path
            )
            return bool(status_out.strip())
        except Exception:
            return False

    def _check_dirty(self, task: WorktreeTask) -> str | None:
        if self._is_dirty(task):
            return f"cannot remove worktree task {task.id}: worktree is dirty (has uncommitted changes)"
        return None

    def _check_unpushed_commits(self, task: WorktreeTask) -> str | None:
        try:
            cherry_out = self._get_cherry_output(task)
            unmerged_commits = [
                line for line in cherry_out.splitlines() if line.startswith("+")
            ]
            if unmerged_commits:
                return f"cannot remove worktree task {task.id}: branch '{task.branch}' has unmerged/unpushed commits"
        except RuntimeError as exc:
            return f"cannot remove worktree task {task.id}: {exc}"
        except Exception:
            logging.warning(
                "git cherry/unmerged check failed in worktree removal",
                exc_info=True,
            )
        return None

    def _get_cherry_output(self, task: WorktreeTask) -> str:
        has_upstream = True
        try:
            self.command_runner(["git", "rev-parse", "--abbrev-ref", "@{u}"], task.path)
        except Exception:
            has_upstream = False

        if has_upstream:
            return self.command_runner(["git", "cherry", "@{u}"], task.path)

        default_branch = self._detect_default_branch(task)
        if default_branch is None:
            raise RuntimeError(
                "no upstream and no default branch; cannot verify unpushed commits — use force=True to bypass"
            )
        return self.command_runner(["git", "cherry", default_branch], task.path)

    def _detect_default_branch(self, task: WorktreeTask) -> str | None:
        """从 origin/HEAD 符号引用或 git remote show origin 推断默认分支。"""
        try:
            ref = self.command_runner(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"], task.path
            )
            head = ref.strip()
            if head.startswith("refs/remotes/origin/"):
                return head[len("refs/remotes/origin/") :]
        except (OSError, RuntimeError, subprocess.SubprocessError):
            logger.debug(
                "git symbolic-ref failed while detecting default branch",
                exc_info=True,
            )
        try:
            out = self.command_runner(["git", "remote", "show", "origin"], task.path)
            for line in out.splitlines():
                if "HEAD branch" in line:
                    return line.split(":")[-1].strip()
        except (OSError, RuntimeError, subprocess.SubprocessError):
            logger.debug(
                "git remote show origin failed while detecting default branch",
                exc_info=True,
            )
        return None


def build_worktree_tools(runner: WorktreeTaskRunner) -> tuple[ToolSpec, ...]:
    def create_worktree_task(args: ToolInput) -> str:
        name = str(args.get("name", "")).strip()
        if not name:
            raise ValueError("name is required")
        task = runner.create(name)
        return f"id={task.id}\nbranch={task.branch}\npath={task.path}"

    def remove_worktree_task(args: ToolInput) -> str:
        task_id = str(args.get("id", "")).strip()
        force = bool(args.get("force", False))
        prune = bool(args.get("prune", False))
        if not task_id:
            raise ValueError("id is required")
        return runner.remove(task_id, force=force, prune=prune)

    def list_worktrees(_args: ToolInput) -> str:
        infos = runner.list()
        if not infos:
            return "No worktrees."
        lines = []
        for info in infos:
            lines.append(
                f"id={info.id}\nbranch={info.branch}\npath={info.path}\n"
                f"dirty={info.dirty}\nexists={info.exists}"
            )
        return "\n".join(lines)

    def prune_stale_worktrees(_args: ToolInput) -> str:
        cleaned = runner.prune_stale()
        if not cleaned:
            return "No stale worktrees to prune."
        lines = [f"Pruned {len(cleaned)} stale worktree(s):"]
        for path in cleaned:
            lines.append(f"  - {path}")
        return "\n".join(lines)

    return (
        ToolSpec(
            "create_worktree_task",
            "Create an isolated git worktree for a task.",
            'JSON: {"name":"feature-name"}',
            create_worktree_task,
            group="worktree",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "remove_worktree_task",
            "Remove an isolated git worktree task. Set prune=true to also clean git worktree metadata.",
            'JSON: {"id":"...", "force":false, "prune":false}',
            remove_worktree_task,
            group="worktree",
            schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "force": {"type": "boolean"},
                    "prune": {"type": "boolean"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "list_worktrees",
            "List all known xcode-managed git worktrees with dirty and existence status.",
            "JSON: {}",
            list_worktrees,
            group="worktree",
            read_only=True,
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "prune_stale_worktrees",
            "Prune orphaned worktree directories under .local/worktrees that are not in the index.",
            "JSON: {}",
            prune_stale_worktrees,
            group="worktree",
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )


def _run_command(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(output or f"command failed: {' '.join(command)}")
    return output
