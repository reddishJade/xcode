from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from xcode.harness.observability import (
    HITLResult,
    PermissionPolicy,
    PermissionRule,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolSpec, run_tool, run_tool_result
from xcode.harness.agent_runtime import StructuredAgent


from xcode.tests.fixtures import FakeProvider
from xcode.ai.events import (
    TextDelta,
    FinalMessage,
    ToolCallEvent,
    ToolCall,
)


INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


class XcodePermissionsTests(unittest.TestCase):
    def test_permission_policy_denies_tool(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda value: value["input"])
        policy = PermissionPolicy((PermissionRule("echo", "deny"),))

        output = run_tool(
            {"echo": tool}, "echo", {"input": "hello"}, permission_policy=policy
        )

        self.assertEqual(output, "permission denied for tool: echo")

    def test_run_tool_result_reports_handler_exception(self) -> None:
        def fail(_value: dict) -> str:
            raise RuntimeError("boom")

        result = run_tool_result(
            {"fail": ToolSpec("fail", "Fail.", "text", fail)}, "fail", {}
        )

        self.assertEqual(result.status, "error")
        self.assertIn("boom", result.content)

    def test_permission_policy_ask_for_low_risk_tool(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda value: value["input"])
        policy = PermissionPolicy((PermissionRule("echo", "ask"),))

        output = run_tool(
            {"echo": tool}, "echo", {"input": "hello"}, permission_policy=policy
        )

        self.assertEqual(output, "tool requires approval: echo")

    def test_permission_policy_allow_skips_high_risk_approval(self) -> None:
        tool = ToolSpec(
            "danger", "Danger.", "text", lambda value: value["input"], risk="high"
        )
        policy = PermissionPolicy((PermissionRule("danger", "allow"),))

        output = run_tool(
            {"danger": tool}, "danger", {"input": "go"}, permission_policy=policy
        )

        self.assertEqual(output, "go")

    def test_structured_agent_uses_permission_policy(self) -> None:
        from xcode.ai.events import ProviderEvent

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="echo", input={"input": "hello"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")],
        ]
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            permission_policy=PermissionPolicy((PermissionRule("echo", "deny"),)),
        )

        result = agent.run("go")

        self.assertIn("permission denied", result.messages[2]["content"][0]["content"])


class HITLPermissionModelTests(unittest.TestCase):
    def test_hitl_result_dataclass(self) -> None:
        r = HITLResult("allow", "once")
        self.assertEqual(r.decision, "allow")
        self.assertEqual(r.scope, "once")

    def test_session_policy_grant_decide(self) -> None:
        policy = SessionPermissionPolicy()
        policy.grant("bash", "allow", "git status")
        self.assertEqual(policy.decide("bash", "git status --short"), "allow")
        self.assertIsNone(policy.decide("bash", "git add ."))

    def test_session_policy_lifo_order(self) -> None:
        policy = SessionPermissionPolicy()
        policy.grant("bash", "deny", "git status")
        policy.grant("bash", "allow", "git status")
        self.assertEqual(policy.decide("bash", "git status"), "allow")

    def test_session_policy_clear(self) -> None:
        policy = SessionPermissionPolicy()
        policy.grant("bash", "allow")
        self.assertIsNotNone(policy.decide("bash", "anything"))
        policy.clear()
        self.assertIsNone(policy.decide("bash", "anything"))

    def test_persistent_store_grant_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentPermissionStore(Path(tmp) / "hitl_policy.json")
            policy = store.grant("bash", "allow", "git status")
            self.assertEqual(policy.decide("bash", "git status"), "allow")
            reloaded = store.load()
            self.assertEqual(reloaded.decide("bash", "git status"), "allow")

    def test_persistent_store_revoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentPermissionStore(Path(tmp) / "hitl_policy.json")
            store.grant("bash", "allow", "git add")
            store.revoke("bash", "git add")
            reloaded = store.load()
            self.assertIsNone(reloaded.decide("bash", "git add"))

    def test_persistent_store_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersistentPermissionStore(Path(tmp) / "nope.json")
            policy = store.load()
            self.assertEqual(policy.rules, ())

    def test_persistent_store_corrupt_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.json"
            path.write_text("not json", encoding="utf-8")
            store = PersistentPermissionStore(path)
            policy = store.load()
            self.assertEqual(policy.rules, ())

    def test_persistent_store_skips_invalid_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    [
                        {"tool": "bash", "decision": "allow"},
                        {"tool": "bash", "decision": "invalid"},
                        {"tool": "write_file", "decision": "deny"},
                    ]
                ),
                encoding="utf-8",
            )
            store = PersistentPermissionStore(path)
            policy = store.load()
            self.assertEqual(len(policy.rules), 2)

    def test_approved_but_failed_handler_sets_approved_true(self) -> None:
        def fail_handler(_value: dict) -> str:
            raise RuntimeError("handler failed")

        tool = ToolSpec(
            "failing",
            "Fails.",
            "text",
            fail_handler,
            risk="high",
        )
        result = run_tool_result(
            {"failing": tool},
            "failing",
            {"input": "go"},
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
        )
        self.assertEqual(result.status, "error")
        self.assertIn("handler failed", result.content)

    def test_denied_callback_returns_guidance_message(self) -> None:
        tool = ToolSpec(
            "bash",
            "Bash.",
            "text",
            lambda v: v["command"],
            risk_evaluator=lambda _: "ask",
        )
        result = run_tool_result(
            {"bash": tool},
            "bash",
            {"command": "git add ."},
            approval_callback=lambda _t, _i: HITLResult("deny", "once"),
        )
        self.assertEqual(result.status, "denied")
        self.assertIn("read-only checks", result.content)

    def test_hitl_result_metadata_in_denied_result(self) -> None:
        tool = ToolSpec(
            "bash",
            "Bash.",
            "text",
            lambda v: v["command"],
            risk_evaluator=lambda _: "ask",
        )
        result = run_tool_result(
            {"bash": tool},
            "bash",
            {"command": "git push"},
            approval_callback=lambda _t, _i: HITLResult("deny", "session"),
        )
        meta = result.metadata or {}
        self.assertEqual(meta.get("user_decision"), "deny")
        self.assertEqual(meta.get("approval_scope"), "session")


if __name__ == "__main__":
    unittest.main()
