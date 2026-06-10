"""托管实现任务使用的 Git worktree 隔离工具。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import uuid

from xcode.harness.skills import ToolInput, ToolSpec


CommandRunner = Callable[[list[str], Path], str]


@dataclass(frozen=True)
class WorktreeTask:
    id: str
    path: Path
    branch: str


class WorktreeTaskRunner:
    def __init__(
        self,
        repo_root: Path,
        worktrees_dir: Path | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.worktrees_dir = worktrees_dir or self.repo_root / ".xcode-worktrees"
        self.command_runner = command_runner or _run_command
        self.tasks: dict[str, WorktreeTask] = {}

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
        self.tasks[task_id] = task
        return task

    def remove(self, task_id: str, force: bool = False) -> str:
        task = self.tasks.get(task_id)
        if task is None:
            return f"unknown worktree task: {task_id}"

        if not force:
            # Check dirty status: modified, staged, or untracked files
            try:
                status_out = self.command_runner(
                    ["git", "status", "--porcelain"], task.path
                )
                if status_out.strip():
                    return f"cannot remove worktree task {task_id}: worktree is dirty (has uncommitted changes)"
            except Exception:
                logging.warning(
                    "git status check failed in worktree removal", exc_info=True
                )
            try:
                has_upstream = True
                try:
                    self.command_runner(
                        ["git", "rev-parse", "--abbrev-ref", "@{u}"], task.path
                    )
                except Exception:
                    has_upstream = False

                if has_upstream:
                    cherry_out = self.command_runner(
                        ["git", "cherry", "@{u}"], task.path
                    )
                else:
                    base_branch: str | None = "main"
                    try:
                        self.command_runner(
                            ["git", "rev-parse", "--verify", "main"], task.path
                        )
                    except Exception:
                        try:
                            self.command_runner(
                                ["git", "rev-parse", "--verify", "master"], task.path
                            )
                            base_branch = "master"
                        except Exception:
                            base_branch = None

                    if base_branch:
                        cherry_out = self.command_runner(
                            ["git", "cherry", base_branch], task.path
                        )
                    else:
                        cherry_out = ""

                unmerged_commits = [
                    line for line in cherry_out.splitlines() if line.startswith("+")
                ]
                if unmerged_commits:
                    return f"cannot remove worktree task {task_id}: branch '{task.branch}' has unmerged/unpushed commits"
            except Exception:
                logging.warning(
                    "git cherry/unmerged check failed in worktree removal",
                    exc_info=True,
                )

        cmd = ["git", "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(str(task.path))

        self.command_runner(cmd, self.repo_root)
        del self.tasks[task_id]
        return f"removed worktree task {task_id}"


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
        if not task_id:
            raise ValueError("id is required")
        return runner.remove(task_id, force=force)

    return (
        ToolSpec(
            "create_worktree_task",
            "Create an isolated git worktree for a task.",
            'JSON: {"name":"feature-name"}',
            create_worktree_task,
            risk="high",
            group="worktree",
        ),
        ToolSpec(
            "remove_worktree_task",
            "Remove an isolated git worktree task.",
            'JSON: {"id":"...", "force":false}',
            remove_worktree_task,
            risk="high",
            group="worktree",
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
