from __future__ import annotations

import unittest

from xcode.harness.observability import (
    HITLResult,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    PermissionRule,
)
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.agent.config import AgentContext, BeforeToolCallContext
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import ToolCallContent


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
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("echo", "deny"),)),
            )
        )
        result = engine.decide(
            "echo", '{"input": "hello"}', tool_spec=tool, tool_input={"input": "hello"}
        )
        self.assertTrue(result.blocked)
        self.assertIn("deny for echo", result.reason)

    def test_handler_exception_reports_error(self) -> None:
        def fail(_value: dict) -> str:
            raise RuntimeError("boom")

        tool = ToolSpec("fail", "Fail.", "text", fail)
        result = PermissionEngine(PermissionEngineConfig()).decide(
            "fail", '{"input": "x"}', tool_spec=tool, tool_input={}
        )
        self.assertFalse(result.blocked)
        with self.assertRaises(RuntimeError):
            tool.handler({})

    def test_permission_policy_ask_for_low_risk_tool(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda value: value["input"])
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("echo", "ask"),)),
            )
        )
        result = engine.decide(
            "echo", '{"input": "hello"}', tool_spec=tool, tool_input={"input": "hello"}
        )
        self.assertTrue(result.blocked)
        self.assertIn("requires approval", result.reason)

    def test_permission_policy_allow_skips_high_risk_approval(self) -> None:
        tool = ToolSpec(
            "danger",
            "Danger.",
            "text",
            lambda value: value["input"],
        )
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("danger", "allow"),)),
            )
        )
        result = engine.decide(
            "danger", '{"input": "go"}', tool_spec=tool, tool_input={"input": "go"}
        )
        self.assertFalse(result.blocked)
        self.assertEqual(tool.handler({"input": "go"}), "go")

    def test_permission_policy_deny_overrides_allow_regardless_of_order(self) -> None:
        rules = (
            PermissionRule("bash", "allow"),
            PermissionRule("*", "deny"),
        )
        policy = PermissionPolicy(rules)
        self.assertEqual(policy.decide("bash", "anything"), "deny")

    def test_permission_engine_restricted_dirs_deny(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                restricted_dirs=("secrets",),
            )
        )
        result = engine.decide("read_file", '{"path": "secrets/key.txt"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.matched_rule, "restricted_dirs")

    def test_permission_engine_session_grant_satisfies_ask(self) -> None:
        from xcode.harness.observability import (
            InMemoryGrantStore,
            ActionExtractor,
            create_grant_record,
        )

        store = InMemoryGrantStore()
        action = ActionExtractor().extract("bash", {"command": "git status"})
        for target in action.targets:
            grant = create_grant_record(
                action, target, decision="allow", scope="session"
            )
            store.add(grant)
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
                session_grant_store=store,
            )
        )
        result = engine.decide(
            "bash",
            '{"command": "git status"}',
            tool_input={"command": "git status"},
        )
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
        # High-risk approval path removed (STEP 5). Default allow.
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("danger", "hello")
        self.assertFalse(result.blocked)

    def test_permission_engine_execution_mode_deny(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("bash", "rm -rf /", execution_decision="deny")
        self.assertTrue(result.blocked)
        self.assertEqual(result.matched_rule, "mode")

    def test_tool_gate_static_deny_preempts_execution_mode_ask(self) -> None:
        called = False

        def approve(_tool: ToolSpec, _input: dict[str, object]) -> HITLResult:
            nonlocal called
            called = True
            return HITLResult("allow", "once")

        mode = ExecutionModeState()
        mode.set_mode("review")
        tool = ToolSpec("bash", "Bash.", "command", lambda _value: "")
        gate = ToolGate(
            mode_state=mode,
            approval_callback=approve,
            permission_policy=PermissionPolicy((PermissionRule("bash", "deny"),)),
            hook_manager=None,
            audit_logger=None,
            session_id="test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x",
                    name="bash",
                    arguments={"command": "git add ."},
                ),
                args={"command": "git add ."},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("deny for bash", result.reason)
        self.assertFalse(called)

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

        self.assertIn("deny for echo", result.messages[2]["content"][0]["content"])

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
                    schema=INPUT_SCHEMA,
                ),
            ),
        )
        agent.approval_callback = lambda _tool, _input: HITLResult("allow", "once")

        result = agent.run("go")

        self.assertEqual(result.messages[2]["content"][0]["content"], "go")

    def test_permission_allows_then_handler_raises(self) -> None:
        def fail_handler(_value: dict) -> str:
            raise RuntimeError("handler failed")

        tool = ToolSpec("failing", "Fails.", "text", fail_handler)
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("failing", "ask"),)),
            )
        )
        result = engine.decide(
            "failing",
            '{"input": "go"}',
            tool_spec=tool,
            tool_input={"input": "go"},
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
        )
        self.assertFalse(result.blocked)
        with self.assertRaises(RuntimeError):
            tool.handler({"input": "go"})

    def test_denied_callback_returns_guidance_message(self) -> None:
        tool = ToolSpec(
            "bash",
            "Bash.",
            "text",
            lambda v: v["command"],
        )
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
            )
        )
        result = engine.decide(
            "bash",
            '{"command": "git add ."}',
            tool_spec=tool,
            tool_input={"command": "git add ."},
            approval_callback=lambda _t, _i: HITLResult("deny", "once"),
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")

    def test_denied_callback_metadata(self) -> None:
        tool = ToolSpec(
            "bash",
            "Bash.",
            "text",
            lambda v: v["command"],
        )
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
            )
        )
        result = engine.decide(
            "bash",
            '{"command": "git push"}',
            tool_spec=tool,
            tool_input={"command": "git push"},
            approval_callback=lambda _t, _i: HITLResult("deny", "session"),
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")


if __name__ == "__main__":
    unittest.main()
