from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from xcode.harness.agent_runtime.events import ToolCall
from xcode.harness.agent_runtime.execution_modes import (
    ActPolicy,
    PlanPolicy,
    ReviewPolicy,
)
from xcode.harness.agent_runtime.tool_executor import (
    ExecutionCancelled,
    ToolExecutor,
    partition_tool_calls,
    tool_result_message,
)
from xcode.cli.repl import ReplHITLHandler
from xcode.harness.observability import (
    AuditRecord,
    HITLResult,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolSpec


class ToolExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_before_execution_raises_without_partial_results(self) -> None:
        cancel = asyncio.Event()
        cancel.set()
        executor = ToolExecutor(
            (ToolSpec("echo", "Echo.", "text", lambda data: data["input"]),)
        )

        with self.assertRaises(ExecutionCancelled):
            await executor.execute(
                [ToolCall("t1", "echo", {"input": "hello"})], cancel=cancel
            )

    async def test_execute_formats_runtime_results(self) -> None:
        executor = ToolExecutor(
            (ToolSpec("echo", "Echo.", "text", lambda data: data["input"]),)
        )

        results = await executor.execute(
            [ToolCall("t1", "echo", {"input": "hello"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(results[0].content, "hello")
        self.assertEqual(
            tool_result_message(results[0]),
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "hello",
                "status": "ok",
            },
        )

    async def test_policy_denies_unavailable_call_at_executor_boundary(self) -> None:
        called = []

        def write_handler(data: dict) -> str:
            called.append(data["input"])
            return data["input"]

        executor = ToolExecutor(
            (
                ToolSpec(
                    "write_file",
                    "Write.",
                    "text",
                    write_handler,
                ),
            ),
            policy=PlanPolicy(),
        )

        results = await executor.execute(
            [ToolCall("t1", "write_file", {"input": "hello"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(results[0].status, "denied")
        self.assertEqual(called, [])

    async def test_require_approval_without_callback_returns_result(self) -> None:
        executor = ToolExecutor(
            (
                ToolSpec(
                    "run_validation",
                    "Run tests.",
                    "command",
                    lambda data: data["input"],
                ),
            ),
            policy=ReviewPolicy(),
        )

        results = await executor.execute(
            [ToolCall("t1", "run_validation", {"input": "tests"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(results[0].status, "approval_required")

    async def test_act_high_risk_tool_still_requires_approval(self) -> None:
        called = []

        def write_handler(data: dict) -> str:
            called.append(data["input"])
            return "wrote"

        executor = ToolExecutor(
            (
                ToolSpec(
                    "write_file",
                    "Write.",
                    "text",
                    write_handler,
                    risk="high",
                ),
            ),
        )

        results = await executor.execute(
            [ToolCall("t1", "write_file", {"input": "content"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(results[0].status, "approval_required")
        self.assertEqual(called, [])

    async def test_act_high_risk_tool_runs_after_user_approval(self) -> None:
        called = []

        def write_handler(data: dict) -> str:
            called.append(data["input"])
            return "wrote"

        executor = ToolExecutor(
            (
                ToolSpec(
                    "write_file",
                    "Write.",
                    "text",
                    write_handler,
                    risk="high",
                ),
            ),
            approval_callback=lambda _tool, _input: HITLResult("allow", "once"),
        )

        results = await executor.execute(
            [ToolCall("t1", "write_file", {"input": "content"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(results[0].status, "ok")
        self.assertEqual(called, ["content"])

    async def test_policy_deny_records_audit(self) -> None:
        audit_records: list[AuditRecord] = []
        executor = ToolExecutor(
            (ToolSpec("write_file", "Write.", "text", lambda value: "ok"),),
            policy=PlanPolicy(),
            audit_logger=lambda r: audit_records.append(r),
        )

        await executor.execute(
            [ToolCall("t1", "write_file", {"input": "content"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(len(audit_records), 1)
        self.assertEqual(audit_records[0].tool, "write_file")
        self.assertEqual(audit_records[0].final_status, "denied")

    async def test_approval_required_records_audit(self) -> None:
        audit_records: list[AuditRecord] = []
        executor = ToolExecutor(
            (ToolSpec("run_validation", "Run tests.", "command", lambda _data: "ok"),),
            policy=ReviewPolicy(),
            audit_logger=lambda r: audit_records.append(r),
        )

        await executor.execute(
            [ToolCall("t1", "run_validation", {"input": "tests"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(len(audit_records), 1)
        self.assertEqual(audit_records[0].final_status, "approval_required")

    async def test_denied_callback_records_user_decision_in_audit(self) -> None:
        audit_records: list[AuditRecord] = []
        executor = ToolExecutor(
            (ToolSpec("run_validation", "Run tests.", "command", lambda value: "ok"),),
            policy=ReviewPolicy(),
            approval_callback=lambda _t, _i: HITLResult("deny", "session"),
            audit_logger=lambda r: audit_records.append(r),
        )

        await executor.execute(
            [ToolCall("t1", "run_validation", {"input": "tests"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(len(audit_records), 1)
        self.assertEqual(audit_records[0].user_decision, "deny")
        self.assertEqual(audit_records[0].approval_scope, "session")

    async def test_allowed_then_handler_error_records_approved_true(self) -> None:
        audit_records: list[AuditRecord] = []

        def fail_handler(_data: dict) -> str:
            raise RuntimeError("fail")

        executor = ToolExecutor(
            (ToolSpec("fail", "Fails.", "text", fail_handler, risk="high"),),
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
            audit_logger=lambda r: audit_records.append(r),
        )

        await executor.execute(
            [ToolCall("t1", "fail", {"input": "x"})],
            cancel=asyncio.Event(),
        )

        self.assertEqual(len(audit_records), 1)
        self.assertEqual(audit_records[0].final_status, "error")
        self.assertTrue(audit_records[0].approved)

    def test_partition_tool_calls(self) -> None:
        # Define some tools
        t_read = ToolSpec(
            "read",
            "Read.",
            "text",
            lambda _data: "",
            read_only=True,
            concurrency_safe=True,
        )
        t_write = ToolSpec(
            "write",
            "Write.",
            "text",
            lambda _data: "",
            read_only=False,
            concurrency_safe=False,
        )
        t_unsafe = ToolSpec(
            "unsafe",
            "Unsafe.",
            "text",
            lambda _data: "",
            read_only=True,
            concurrency_safe=False,
        )
        t_high_risk = ToolSpec(
            "high_risk",
            "High risk.",
            "text",
            lambda _data: "",
            read_only=True,
            concurrency_safe=True,
            risk="high",
        )

        active_map = {t.name: t for t in (t_read, t_write, t_unsafe, t_high_risk)}

        calls = [
            ToolCall("c1", "read", {"input": "1"}),
            ToolCall("c2", "read", {"input": "2"}),
            ToolCall("c3", "write", {"input": "3"}),
            ToolCall("c4", "read", {"input": "4"}),
            ToolCall("c5", "unsafe", {"input": "5"}),
            ToolCall("c6", "high_risk", {"input": "6"}),
        ]

        batches = partition_tool_calls(calls, active_map)

        self.assertEqual(len(batches), 5)
        self.assertEqual([c.id for c in batches[0]], ["c1", "c2"])
        self.assertEqual([c.id for c in batches[1]], ["c3"])
        self.assertEqual([c.id for c in batches[2]], ["c4"])
        self.assertEqual([c.id for c in batches[3]], ["c5"])
        self.assertEqual([c.id for c in batches[4]], ["c6"])


class ExecutionModeTests(unittest.TestCase):
    def test_act_validation_requires_approval(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(ToolCall("t1", "run_validation", {}))
        self.assertEqual(result, "require_approval")

    def test_act_bash_still_allowed(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(ToolCall("t1", "bash", {"command": "echo hello"}))
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
                patch("xcode.cli.repl._has_radiolist", return_value=True),
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
