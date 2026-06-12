from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from xcode.harness.observability import (
    HITLResult,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    PermissionRule,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolSpec, run_tool_result
from xcode.tests.fixtures import run_tool
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig


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

    def test_permission_policy_deny_overrides_allow_regardless_of_order(self) -> None:
        rules = (
            PermissionRule("bash", "allow"),
            PermissionRule("*", "deny"),
        )
        policy = PermissionPolicy(rules)
        self.assertEqual(policy.decide("bash", "anything"), "deny")

    def test_permission_engine_static_deny_is_final(self) -> None:
        session = SessionPermissionPolicy()
        session.grant("echo", "allow")
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("echo", "deny"),)),
                session_policy=session,
            )
        )
        result = engine.decide("echo", "hello")
        self.assertTrue(result.blocked)
        self.assertEqual(result.matched_rule, "static_deny")

    def test_permission_engine_restricted_dirs_deny(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                restricted_dirs=("secrets",),
            )
        )
        result = engine.decide("read_file", '{"path": "secrets/key.txt"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.matched_rule, "restricted_dirs")

    def test_permission_engine_restricted_dirs_override_static_allow(self) -> None:
        session = SessionPermissionPolicy()
        session.grant("read_file", "allow", "secrets/key.txt")
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("read_file", "allow"),)),
                restricted_dirs=("secrets",),
                session_policy=session,
            )
        )
        result = engine.decide("read_file", '{"path": "secrets/key.txt"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.matched_rule, "restricted_dirs")

    def test_permission_engine_session_grant_satisfies_ask(self) -> None:
        session = SessionPermissionPolicy()
        session.grant("bash", "allow", "git status")
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
                session_policy=session,
            )
        )
        result = engine.decide("bash", "git status --short")
        self.assertFalse(result.blocked)
        self.assertEqual(result.matched_rule, "session_grant")

    def test_permission_engine_allowlist_mode(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("read_file", "allow"),)),
                allowlist_mode=True,
            )
        )
        allowed = engine.decide("read_file", '{"path": "a.txt"}')
        self.assertFalse(allowed.blocked)
        unknown = engine.decide("write_file", '{"path": "b.txt"}')
        self.assertTrue(unknown.blocked)
        self.assertEqual(unknown.decision, "ask")

    def test_permission_engine_default_allow(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("any_tool", "anything")
        self.assertFalse(result.blocked)
        self.assertEqual(result.matched_rule, "default")

    def test_permission_engine_high_risk_approval(self) -> None:
        tool = ToolSpec("danger", "Danger.", "text", lambda v: v["input"], risk="high")
        engine = PermissionEngine(
            PermissionEngineConfig(high_risk_requires_approval=True)
        )
        result = engine.decide(
            "danger",
            "hello",
            tool_spec=tool,
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
        )
        self.assertFalse(result.blocked)

    def test_permission_engine_execution_mode_deny(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("bash", "rm -rf /", execution_decision="deny")
        self.assertTrue(result.blocked)
        self.assertEqual(result.matched_rule, "execution_mode")

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
            gate=GateConfig(
                permission_policy=PermissionPolicy((PermissionRule("echo", "deny"),)),
            ),
        )

        result = agent.run("go")

        self.assertIn("permission denied", result.messages[2]["content"][0]["content"])

    def test_structured_agent_approval_callback_setter_updates_gate(self) -> None:
        from xcode.ai.events import ProviderEvent

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="danger", input={"input": "go"})]
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
                    "danger",
                    "Danger.",
                    "text",
                    lambda value: value["input"],
                    risk="high",
                    schema=INPUT_SCHEMA,
                ),
            ),
        )
        agent.approval_callback = lambda _tool, _input: HITLResult("allow", "once")

        result = agent.run("go")

        self.assertEqual(result.messages[2]["content"][0]["content"], "go")


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
