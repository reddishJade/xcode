from __future__ import annotations

import tempfile
from pathlib import Path

from xcode.coding_agent.tools.worktree import WorktreeTaskRunner
import pytest
class TestWorktreeTaskRunner:
    def setup_method(self, method) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmp.name) / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        self.worktrees_dir = self.repo_root / ".local" / "worktrees"
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self.commands_run: list[tuple[list[str], Path]] = []
        self.mock_responses: dict[str, str | Exception] = {}

    def mock_command_runner(self, cmd: list[str], cwd: Path) -> str:
        self.commands_run.append((cmd, cwd))
        cmd_str = " ".join(cmd)

        for pattern, response in self.mock_responses.items():
            if pattern in cmd_str:
                if isinstance(response, Exception):
                    raise response
                return response
        return ""

    def teardown_method(self, method) -> None:
        self.tmp.cleanup()

    def test_create_worktree_task(self) -> None:
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )

        task = runner.create("feature-login")

        assert len(runner.tasks) == 1
        assert task.branch[:14] == "xcode/feature-"
        assert task.path.is_relative_to(self.worktrees_dir)

        # Check command execution
        assert len(self.commands_run) == 1
        cmd, cwd = self.commands_run[0]
        assert cmd[0:3] == ["git", "worktree", "add"]
        assert cwd == self.repo_root

    def test_remove_clean_happy_path(self) -> None:
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("clean-task")
        self.commands_run.clear()

        # Mock clean status
        self.mock_responses["git status --porcelain"] = ""
        # Mock upstream check failing (no upstream branch)
        self.mock_responses["git rev-parse --abbrev-ref @{u}"] = Exception(
            "no upstream"
        )
        # Mock verifying main branch succeeding
        self.mock_responses["git rev-parse --verify main"] = "main_hash"
        # Mock git cherry main returning no unmerged commits
        self.mock_responses["git cherry main"] = ""

        res = runner.remove(task.id)
        assert res == f"removed worktree task {task.id}"
        assert len(runner.tasks) == 0

        # Check git worktree remove command was run
        remove_cmd_exists = any(
            "git worktree remove" in " ".join(cmd) for cmd, _ in self.commands_run
        )
        assert remove_cmd_exists

    def test_remove_dirty_blocker(self) -> None:
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("dirty-task")
        self.commands_run.clear()

        # Mock dirty status output
        self.mock_responses["git status --porcelain"] = " M file.py\n?? untracked.py\n"

        res = runner.remove(task.id)
        assert "worktree is dirty (has uncommitted changes)" in res
        assert len(runner.tasks) == 1

        # Ensure git worktree remove was never run
        remove_cmd_exists = any(
            "git worktree remove" in " ".join(cmd) for cmd, _ in self.commands_run
        )
        assert not (remove_cmd_exists)

    def test_remove_unmerged_cherry_blocker(self) -> None:
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("cherry-task")
        self.commands_run.clear()

        self.mock_responses["git status --porcelain"] = ""
        # Upstream exists
        self.mock_responses["git rev-parse --abbrev-ref @{u}"] = "origin/task-branch"
        # Has 1 unmerged/unpushed commit
        self.mock_responses["git cherry"] = "+ abc123commitsha\n"

        res = runner.remove(task.id)
        assert "has unmerged/unpushed commits" in res
        assert len(runner.tasks) == 1

        # Ensure git worktree remove was never run
        remove_cmd_exists = any(
            "git worktree remove" in " ".join(cmd) for cmd, _ in self.commands_run
        )
        assert not (remove_cmd_exists)

    def test_remove_force_bypass(self) -> None:
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("force-task")
        self.commands_run.clear()

        res = runner.remove(task.id, force=True)
        assert res == f"removed worktree task {task.id}"
        assert len(runner.tasks) == 0

        # Verify only 'git worktree remove --force' command was run
        assert len(self.commands_run) == 1
        cmd, cwd = self.commands_run[0]
        assert "remove" in cmd
        assert "--force" in cmd
        assert cwd == self.repo_root

if __name__ == "__main__":
    pytest.main()
