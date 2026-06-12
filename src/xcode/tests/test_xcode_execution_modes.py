from __future__ import annotations

import asyncio
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from xcode.ai.events import ToolCall
from xcode.cli.repl_hitl import ReplHITLHandler
from xcode.harness.agent_runtime.execution_modes import ActPolicy
from xcode.harness.observability import (
    HITLResult,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolSpec


class ExecutionModeTests(unittest.TestCase):
    def test_act_bash_still_allowed(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="bash", input={"command": "echo hello"})
        )
        self.assertEqual(result, "allow")


class ReplHITLHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_policy = SessionPermissionPolicy()
        self.persistent_store = PersistentPermissionStore(Path(""))
        self.handler = ReplHITLHandler(self.session_policy, self.persistent_store)
        self.tool = ToolSpec("bash", "Bash.", "text", lambda _data: "")

    def test_handler_allow_once(self) -> None:
        result = self.handler._apply_choice(
            "允许（仅本次）", self.tool, {"command": "echo hello"}
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "once")

    def test_handler_session_scope(self) -> None:
        result = self.handler._apply_choice(
            "此次对话中允许", self.tool, {"command": "echo hello"}
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "session")
        self.assertIsNotNone(
            self.session_policy.decide("bash", '{"command": "echo hello"}')
        )

    def test_handler_session_scope_uses_validation_command_prefix(self) -> None:
        result = self.handler._apply_choice(
            "此次对话中允许",
            self.tool,
            {"command": "uv run pyright src/xcode/a.py"},
        )

        self.assertEqual(result.decision, "allow")
        self.assertEqual(
            self.session_policy.decide(
                "bash", '{"command": "uv run pyright src/xcode/b.py"}'
            ),
            "allow",
        )

    def test_handler_permanent_scope(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentPermissionStore(Path(tmp) / "hitl.json")
            handler = ReplHITLHandler(SessionPermissionPolicy(), store)
            result = handler._apply_choice(
                "始终允许", self.tool, {"command": "git push"}
            )
            self.assertEqual(result.decision, "allow")
            self.assertEqual(result.scope, "permanent")
            loaded = store.load()
            self.assertIsNotNone(loaded.decide("bash", '{"command": "git push"}'))

    def test_handler_permanent_scope_uses_validation_command_prefix(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentPermissionStore(Path(tmp) / "hitl.json")
            handler = ReplHITLHandler(SessionPermissionPolicy(), store)
            result = handler._apply_choice(
                "始终允许",
                self.tool,
                {"command": "uv run pyright src/xcode/a.py"},
            )

            self.assertEqual(result.decision, "allow")
            self.assertEqual(result.scope, "permanent")
            loaded = store.load()
            self.assertEqual(
                loaded.decide("bash", '{"command": "uv run pyright src/xcode/b.py"}'),
                "allow",
            )

    def test_handler_deny(self) -> None:
        result = self.handler._apply_choice("拒绝", self.tool, {"command": "git add ."})
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.scope, "once")

    def test_unknown_choice_treated_as_deny(self) -> None:
        result = self.handler._apply_choice(None, self.tool, {})
        self.assertEqual(result.decision, "deny")

    def test_session_policy_auto_allows_within_session(self) -> None:
        self.session_policy.grant("bash", "allow", "git commit")
        result = self.handler(self.tool, {"command": "git commit -m 'fix'"})
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "session")

    def test_persistent_policy_auto_allows(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            persistent_store = PersistentPermissionStore(Path(tmp) / "hitl.json")
            persistent_store.grant("bash", "allow", "git push")
            handler = ReplHITLHandler(SessionPermissionPolicy(), persistent_store)
            result = handler(self.tool, {"command": "git push origin main"})
            self.assertEqual(result.decision, "allow")
            self.assertEqual(result.scope, "permanent")

    def test_interactive_prompt_works_inside_running_event_loop(self) -> None:
        question = Mock()
        question.ask.return_value = "允许（仅本次）"

        async def run_prompt() -> HITLResult:
            return self.handler._interactive_prompt(
                self.tool, {"command": "rm -rf /tmp/xcode-demo"}
            )

        with patch("questionary.select", return_value=question):
            result = asyncio.run(run_prompt())

        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "once")
        question.ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
