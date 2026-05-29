from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xcode.experimental.worktree import WorktreeTaskRunner, build_worktree_tools


class XcodeWorktreeTests(unittest.TestCase):
    def test_worktree_runner_builds_git_commands(self) -> None:
        commands = []

        def fake_runner(command: list[str], cwd: Path) -> str:
            commands.append((command, cwd))
            if any(x in command for x in ("status", "cherry", "rev-parse")):
                return ""
            return "ok"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = WorktreeTaskRunner(root, command_runner=fake_runner)

            task = runner.create("demo task")
            removed = runner.remove(task.id)

        self.assertEqual(commands[0][0][:3], ["git", "worktree", "add"])
        remove_cmds = [
            cmd[0] for cmd, _ in commands if cmd[:3] == ["git", "worktree", "remove"]
        ]
        self.assertEqual(len(remove_cmds), 1)
        self.assertIn("removed worktree task", removed)

    def test_worktree_tools_are_high_risk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = build_worktree_tools(
                WorktreeTaskRunner(
                    Path(temp_dir), command_runner=lambda _cmd, _cwd: "ok"
                )
            )

        self.assertEqual(tools[0].risk, "high")
        self.assertEqual(tools[1].risk, "high")


if __name__ == "__main__":
    unittest.main()
