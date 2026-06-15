from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from xcode.ai.events import ToolCall
from xcode.cli.repl_hitl import ReplHITLHandler
from xcode.harness.agent_runtime.execution_modes import ActPolicy, BuildPolicy
from xcode.harness.observability import (
    ActionExtractor,
    FileGrantStore,
    HITLResult,
    InMemoryGrantStore,
    create_grant_record,
)
from xcode.harness.skills import ToolSpec


class ExecutionModeTests(unittest.TestCase):
    def test_act_bash_still_allowed(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="bash", input={"command": "echo hello"})
        )
        self.assertEqual(result, "allow")

    def test_build_policy_allows_file_write(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(
                id="t1", name="write_file", input={"path": "foo.txt", "content": "x"}
            )
        )
        self.assertEqual(result, "allow")

    def test_build_policy_allows_bash(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="bash", input={"command": "echo hello"})
        )
        self.assertEqual(result, "allow")

    def test_build_policy_denies_network_tool(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="curl", input={"url": "http://example.com"})
        )
        self.assertEqual(result, "deny")


class ReplHITLHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_store = InMemoryGrantStore()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.permanent_store = FileGrantStore(Path(self.tmpdir.name) / "grants.json")
        self.handler = ReplHITLHandler(self.session_store, self.permanent_store)
        self.tool = ToolSpec("bash", "Bash.", "text", lambda _data: "")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _action(self, command: str):
        return ActionExtractor().extract("bash", {"command": command})

    def test_handler_allow_once(self) -> None:
        result = self.handler._apply_choice(
            "允许（仅本次）",
            self._action("echo hello"),
            self.tool,
            {"command": "echo hello"},
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "once")

    def test_handler_session_scope(self) -> None:
        result = self.handler._apply_choice(
            "此次对话中允许",
            self._action("echo hello"),
            self.tool,
            {"command": "echo hello"},
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "session")
        # 验证 session store 已有记录
        action = self._action("echo hello")
        for target in action.targets:
            self.assertIsNotNone(self.session_store.lookup(action, target))

    def test_handler_permanent_scope(self) -> None:
        result = self.handler._apply_choice(
            "始终允许", self._action("git push"), self.tool, {"command": "git push"}
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "permanent")
        action = self._action("git push")
        for target in action.targets:
            self.assertIsNotNone(self.permanent_store.lookup(action, target))

    def test_handler_deny(self) -> None:
        result = self.handler._apply_choice(
            "拒绝", self._action("git add ."), self.tool, {"command": "git add ."}
        )
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.scope, "once")

    def test_unknown_choice_treated_as_deny(self) -> None:
        result = self.handler._apply_choice(None, self._action(""), self.tool, {})
        self.assertEqual(result.decision, "deny")

    def test_session_grant_auto_allows_within_session(self) -> None:
        action = self._action("git commit -m 'fix'")
        for target in action.targets:
            grant = create_grant_record(
                action, target, decision="allow", scope="session"
            )
            self.session_store.add(grant)
        result = self.handler(self.tool, {"command": "git commit -m 'fix'"})
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "session")

    def test_permanent_grant_auto_allows(self) -> None:
        action = self._action("git push origin main")
        for target in action.targets:
            grant = create_grant_record(
                action, target, decision="allow", scope="permanent"
            )
            self.permanent_store.add(grant)
        result = self.handler(self.tool, {"command": "git push origin main"})
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "permanent")

    def test_interactive_prompt_works_inside_running_event_loop(self) -> None:
        question = Mock()
        question.ask.return_value = "允许（仅本次）"

        async def run_prompt() -> HITLResult:
            action = ActionExtractor().extract(
                "bash", {"command": "rm -rf /tmp/xcode-demo"}
            )
            return self.handler._interactive_prompt(
                action, self.tool, {"command": "rm -rf /tmp/xcode-demo"}
            )

        with patch("questionary.select", return_value=question):
            result = asyncio.run(run_prompt())

        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "once")
        question.ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
