from __future__ import annotations

import json
import tempfile
import threading
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

        # __init__ reconcile 跑一次 git worktree list，create 跑一次 git worktree add
        assert len(self.commands_run) == 2
        add_cmd, _ = self.commands_run[1]
        assert add_cmd[0:3] == ["git", "worktree", "add"]

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
        self.mock_responses["git rev-parse --abbrev-ref @{u}"] = RuntimeError(
            "no upstream"
        )
        # Mock default branch detection via symbolic-ref
        self.mock_responses["git symbolic-ref refs/remotes/origin/HEAD"] = (
            "refs/remotes/origin/main"
        )
        # Mock git cherry main returning no unmerged commits
        self.mock_responses["git cherry main"] = ""

        res = runner.remove(task.id)
        assert res == f"removed worktree task {task.id}"
        assert len(runner.tasks) == 0

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

        self.mock_responses["git status --porcelain"] = " M file.py\n?? untracked.py\n"

        res = runner.remove(task.id)
        assert "worktree is dirty (has uncommitted changes)" in res
        assert len(runner.tasks) == 1

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
        self.mock_responses["git rev-parse --abbrev-ref @{u}"] = "origin/task-branch"
        self.mock_responses["git cherry"] = "+ abc123commitsha\n"

        res = runner.remove(task.id)
        assert "has unmerged/unpushed commits" in res
        assert len(runner.tasks) == 1

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

        # __init__ reconcile 之后清空了 commands_run，force remove 只跑一条
        assert len(self.commands_run) == 1
        cmd, _ = self.commands_run[0]
        assert "remove" in cmd
        assert "--force" in cmd

    def test_remove_with_prune_calls_git_prune(self) -> None:
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("prune-task")
        self.commands_run.clear()

        res = runner.remove(task.id, force=True, prune=True)
        assert res == f"removed worktree task {task.id}"
        prune_called = any(
            "git worktree prune" in " ".join(cmd) for cmd, _ in self.commands_run
        )
        assert prune_called

    def test_persist_worktree_index(self) -> None:
        """进程重启后可从 index.json 恢复 worktree 列表。"""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("persist-task")
        index_path = self.worktrees_dir / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text(encoding="utf-8"))
        assert any(w["id"] == task.id for w in data["worktrees"])

        # 模拟重启：新建 runner；git worktree list 仍报告该 worktree
        self.mock_responses["git worktree list --porcelain"] = (
            f"worktree {task.path}\nbranch {task.branch}\n"
        )
        runner2 = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        assert task.id in runner2.tasks
        assert runner2.tasks[task.id].branch == task.branch

    def test_reconcile_drops_stale_index_entry(self) -> None:
        """git worktree list 中不再存在的条目，重启后从 index 移除。"""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("stale-task")
        # 手动从 git 移除（模拟外部清理），但 index 仍有条目
        # 通过让 git worktree list 只返回主仓库来模拟 worktree 已不在 git 中
        self.mock_responses["git worktree list --porcelain"] = (
            f"worktree {self.repo_root}\nbranch main\n"
        )
        runner2 = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        assert task.id not in runner2.tasks

    def test_list_worktrees_returns_info(self) -> None:
        """list() 返回 WorktreeInfo 含 dirty/exists。"""
        self.mock_responses["git worktree list --porcelain"] = ""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("list-task")
        # 让 list 时 git 知道这个 worktree
        self.mock_responses["git worktree list --porcelain"] = (
            f"worktree {task.path}\nbranch {task.branch}\n"
        )
        self.mock_responses["git status --porcelain"] = ""
        infos = runner.list()
        assert len(infos) == 1
        assert infos[0].id == task.id
        assert infos[0].exists is True
        assert infos[0].dirty is False

    def test_list_worktrees_dirty_detection(self) -> None:
        """list() 检测 dirty 状态。"""
        self.mock_responses["git worktree list --porcelain"] = ""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("dirty-list-task")
        self.mock_responses["git worktree list --porcelain"] = (
            f"worktree {task.path}\nbranch {task.branch}\n"
        )
        self.mock_responses["git status --porcelain"] = " M modified.py\n"
        infos = runner.list()
        assert infos[0].dirty is True

    def test_list_worktrees_not_in_git_marks_exists_false(self) -> None:
        """git 不再注册的 worktree 标记 exists=False。"""
        self.mock_responses["git worktree list --porcelain"] = ""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        runner.create("gone-task")
        # list 时 git worktree list 返回空（不含该 worktree）
        self.mock_responses["git worktree list --porcelain"] = ""
        infos = runner.list()
        assert len(infos) == 1
        assert infos[0].exists is False

    def test_remove_no_remote_errors_explicitly(self) -> None:
        """无 upstream 且无默认分支时返回明确错误而非静默通过。"""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("no-remote-task")
        self.commands_run.clear()

        self.mock_responses["git status --porcelain"] = ""
        self.mock_responses["git rev-parse --abbrev-ref @{u}"] = RuntimeError(
            "no upstream"
        )
        # symbolic-ref 和 remote show origin 都失败
        self.mock_responses["git symbolic-ref"] = RuntimeError("no head ref")
        self.mock_responses["git remote show origin"] = RuntimeError("no remote")

        res = runner.remove(task.id)
        assert "no upstream and no default branch" in res
        assert len(runner.tasks) == 1  # 未被移除

    def test_default_branch_detection_does_not_hide_programming_errors(self) -> None:
        """默认分支探测不得吞掉非命令执行类异常。"""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("broken-runner-task")
        self.mock_responses["git symbolic-ref"] = ValueError("broken runner")

        with pytest.raises(ValueError, match="broken runner"):
            runner._detect_default_branch(task)

    def test_remove_detects_custom_default_branch(self) -> None:
        """默认分支为 develop 时也能检测未推送提交。"""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        task = runner.create("dev-branch-task")
        self.commands_run.clear()

        self.mock_responses["git status --porcelain"] = ""
        self.mock_responses["git rev-parse --abbrev-ref @{u}"] = RuntimeError(
            "no upstream"
        )
        self.mock_responses["git symbolic-ref refs/remotes/origin/HEAD"] = (
            "refs/remotes/origin/develop"
        )
        self.mock_responses["git cherry develop"] = "+ abc123\n"

        res = runner.remove(task.id)
        assert "has unmerged/unpushed commits" in res
        assert len(runner.tasks) == 1

    def test_prune_stale_removes_orphan_dirs(self) -> None:
        """prune_stale 清理 index 外的孤儿 worktree 目录。"""
        self.mock_responses["git worktree list --porcelain"] = ""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        # 构造一个孤儿目录，模拟是 git worktree
        orphan = self.worktrees_dir / "orphan1"
        orphan.mkdir(parents=True)
        # mock 确认它是 worktree
        self.mock_responses["git rev-parse --is-inside-worktree"] = "true"
        self.mock_responses["git worktree remove --force"] = ""

        cleaned = runner.prune_stale()
        assert orphan in cleaned

    def test_prune_does_not_touch_non_worktree_dirs(self) -> None:
        """prune_stale 跳过非 git worktree 的目录。"""
        self.mock_responses["git worktree list --porcelain"] = ""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )
        junk = self.worktrees_dir / "junk"
        junk.mkdir(parents=True)
        (junk / "file.txt").write_text("data", encoding="utf-8")
        # mock 说它不是 worktree
        self.mock_responses["git rev-parse --is-inside-worktree"] = "false"

        cleaned = runner.prune_stale()
        assert junk not in cleaned
        assert junk.exists()  # 保留

    def test_concurrent_create_safe(self) -> None:
        """并发 create 不损坏 index.json。"""
        self.mock_responses["git worktree list --porcelain"] = ""
        runner = WorktreeTaskRunner(
            repo_root=self.repo_root,
            worktrees_dir=self.worktrees_dir,
            command_runner=self.mock_command_runner,
        )

        def create(name: str) -> str:
            return runner.create(name).id

        threads = [threading.Thread(target=create, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(runner.tasks) == 5
        # index.json 可正常解析且含 5 条
        data = json.loads(
            (self.worktrees_dir / "index.json").read_text(encoding="utf-8")
        )
        assert len(data["worktrees"]) == 5


if __name__ == "__main__":
    pytest.main()
