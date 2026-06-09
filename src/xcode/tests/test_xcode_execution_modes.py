from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

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
    def test_act_validation_requires_approval(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(ToolCall(id="t1", name="run_validation", input={}))
        self.assertEqual(result, "ask")

    def test_act_bash_still_allowed(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(ToolCall(id="t1", name="bash", input={"command": "echo hello"}))
        self.assertEqual(result, "allow")


class ReplHITLHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_policy = SessionPermissionPolicy()
        self.persistent_store = PersistentPermissionStore(Path(""))
        self.handler = ReplHITLHandler(self.session_policy, self.persistent_store)
        self.tool = ToolSpec("bash", "Bash.", "text", lambda _data: "")

    def test_handler_allow_once(self) -> None:
        result = self.handler._apply_choice("1", self.tool, {"command": "echo hello"})
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "once")

    def test_handler_session_scope(self) -> None:
        result = self.handler._apply_choice("2", self.tool, {"command": "echo hello"})
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "session")
        self.assertIsNotNone(
            self.session_policy.decide("bash", '{"command": "echo hello"}')
        )

    def test_handler_permanent_scope(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentPermissionStore(Path(tmp) / "hitl.json")
            handler = ReplHITLHandler(SessionPermissionPolicy(), store)
            result = handler._apply_choice("3", self.tool, {"command": "git push"})
            self.assertEqual(result.decision, "allow")
            self.assertEqual(result.scope, "permanent")
            loaded = store.load()
            self.assertIsNotNone(loaded.decide("bash", '{"command": "git push"}'))

    def test_handler_deny(self) -> None:
        result = self.handler._apply_choice("4", self.tool, {"command": "git add ."})
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.scope, "once")

    def test_unknown_choice_treated_as_deny(self) -> None:
        result = self.handler._apply_choice("x", self.tool, {})
        self.assertEqual(result.decision, "deny")

    def test_async_context_uses_plain_input_not_radiolist(self) -> None:
        import builtins
        from unittest.mock import patch

        async def main() -> HITLResult:
            with (
                patch("xcode.cli.repl_hitl.has_radiolist", return_value=True),
                patch.object(
                    builtins,
                    "input",
                    return_value="1",
                ) as mock_input,
            ):
                result = self.handler(self.tool, {"command": "echo hello"})
                self.assertTrue(mock_input.called)
                prompt_text = mock_input.call_args.args[0]
                self.assertTrue(prompt_text.startswith("\r\033[K\n"))
                self.assertIn("approve [1-4]> ", prompt_text)
                return result

        result = asyncio.run(main())
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.scope, "once")

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


if __name__ == "__main__":
    unittest.main()
