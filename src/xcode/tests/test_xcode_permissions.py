from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xcode.harness.observability import (
    HITLResult,
    HookManager,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    PermissionRule,
)
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
from xcode.harness.agent_runtime.tool_gate import ToolGate, ToolGateSnapshot
from xcode.agent.config import AgentContext, BeforeToolCallContext, BeforeToolCallResult
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


# ── Subagent Permission Boundary Tests ──


class SubagentGatePermissionBoundaryTests(unittest.TestCase):
    """验证子代理 ToolGate 的权限边界行为。

    子代理 GateConfig 不含 approval_callback，
    但应继承 SecurityRuntimeConfig 的非交互式策略。
    """

    def test_subagent_dangerous_shell_blocked_by_safety_backstop(self) -> None:
        """Bucket A 命令即使无静态策略也被 SafetyBackstop 阻断。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "rm -rf /"}
                ),
                args={"command": "rm -rf /"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_subagent_bucket_b_blocked_no_approval_callback(self) -> None:
        """Bucket B ask 命令因无 approval_callback 被阻断。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "rm old_file.txt"}
                ),
                args={"command": "rm old_file.txt"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_subagent_allowed_shell_runs(self) -> None:
        """Bucket C 已知安全命令通过。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "ls -la"}
                ),
                args={"command": "ls -la"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNone(result)

    def test_subagent_static_deny_inherited(self) -> None:
        """子代理继承 deny_tools 静态策略。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy((PermissionRule("bash", "deny"),)),
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "ls"}
                ),
                args={"command": "ls"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_subagent_allowlist_mode(self) -> None:
        """子代理继承 allowlist_mode，仅 allowlist 工具可通过。"""
        mode = ExecutionModeState()
        read_tool = ToolSpec(
            "read_file", "Read.", "path", lambda v: "", schema=INPUT_SCHEMA
        )
        write_tool = ToolSpec(
            "write_file", "Write.", "path", lambda v: "", schema=INPUT_SCHEMA
        )
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy((PermissionRule("read_file", "allow"),)),
            allowlist_mode=True,
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        gate_snapshot = gate.snapshot_for((read_tool, write_tool))
        hook = gate.build_before_tool_hook(gate_snapshot)

        # allowlist 中的 read_file 可通过
        allowed = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="read_file", arguments={"path": "a.txt"}
                ),
                args={"path": "a.txt"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNone(allowed)

        # 不在 allowlist 中的 write_file 被阻断
        blocked = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="y", name="write_file", arguments={"path": "b.txt"}
                ),
                args={"path": "b.txt"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(blocked)
        assert blocked is not None
        self.assertTrue(blocked.block)

    def test_subagent_restricted_dirs_blocks_path(self) -> None:
        """子代理继承 restricted_dirs，匹配路径被阻断。"""
        mode = ExecutionModeState()
        tool = ToolSpec("read_file", "Read.", "path", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            restricted_dirs=("secrets",),
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="read_file", arguments={"path": "secrets/key.txt"}
                ),
                args={"path": "secrets/key.txt"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_subagent_approval_callback_not_called(self) -> None:
        """子代理无 approval_callback，ask 命令不触发回调。"""
        called: list[bool] = []

        def callback(_tool: ToolSpec, _input: dict[str, object]) -> HITLResult:
            called.append(True)
            return HITLResult("allow", "once")

        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,  # 子代理：无回调
            permission_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "git push"}
                ),
                args={"command": "git push"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)
        self.assertEqual(len(called), 0)

    def test_subagent_no_session_grant_store(self) -> None:
        """子代理无 session grant store，ask 不能被已存在的 grant 满足。"""
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

        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))

        # 即使全局 grant store 有匹配记录，子代理 gate 无 store 仍被阻断
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "git status"}
                ),
                args={"command": "git status"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_subagent_permanent_grant_ignored(self) -> None:
        """子代理无 permanent grant store，永久授权不生效。"""
        from xcode.harness.observability import (
            FileGrantStore,
            ActionExtractor,
            create_grant_record,
        )

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            store = FileGrantStore(Path(f.name))
        try:
            action = ActionExtractor().extract("bash", {"command": "git status"})
            for target in action.targets:
                grant = create_grant_record(
                    action, target, decision="allow", scope="permanent"
                )
                store.add(grant)

            mode = ExecutionModeState()
            tool = ToolSpec(
                "bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA
            )
            gate = ToolGate(
                mode_state=mode,
                approval_callback=None,
                permission_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
                hook_manager=None,
                audit_logger=None,
                session_id="subagent-test",
            )
            hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
            result = hook(
                BeforeToolCallContext(
                    assistant_message=AssistantMessage(content=[]),
                    tool_call=ToolCallContent(
                        id="x", name="bash", arguments={"command": "git status"}
                    ),
                    args={"command": "git status"},
                    context=AgentContext(),
                ),
                None,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.block)
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_subagent_structured_agent_with_static_deny(self) -> None:
        """子代理 StructuredAgent 使用带 deny 策略的 GateConfig。"""
        from xcode.ai.events import ProviderEvent

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="echo", input={"input": "hello"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
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

    def test_subagent_structured_agent_audit_emitted(self) -> None:
        """子代理 StructuredAgent 的 GateConfig 带 audit_logger 时发出审计。"""
        from xcode.ai.events import ProviderEvent

        audit_records: list[object] = []

        def capture_audit(record: object) -> None:
            audit_records.append(record)

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="echo", input={"input": "hello"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
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
                audit_logger=capture_audit,
            ),
        )
        agent.run("go")
        self.assertGreater(len(audit_records), 0)

    def test_subagent_structured_agent_hook_manager_wired(self) -> None:
        """子代理 GateConfig 带 hook_manager 时发出 hook 事件。"""
        from xcode.ai.events import ProviderEvent

        hook_records: list[object] = []

        def capture_hook(record: object) -> None:
            hook_records.append(record)

        hook_manager = HookManager()
        hook_manager.register("pre_tool", capture_hook)

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="echo", input={"input": "hello"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
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
                hook_manager=hook_manager,
            ),
        )
        agent.run("go")
        self.assertGreater(len(hook_records), 0)


# ── Project Root / Boundary Resolution Tests ──


class ToolGateBoundaryResolutionTests(unittest.TestCase):
    """验证 project_root 传送到 StructuredBoundaryPolicyEvaluator 后
    能正确解析工作区相对路径。"""

    PATH_SCHEMA = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._root = Path(self._tmp).resolve()
        (self._root / "subdir").mkdir()
        (self._root / "secrets").mkdir()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_file_gate(
        self, project_root: Path | None = None
    ) -> tuple[ToolGate, ToolGateSnapshot]:
        mode = ExecutionModeState()
        tool = ToolSpec(
            "write_file", "Write.", "path", lambda v: "", schema=self.PATH_SCHEMA
        )
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="boundary-test",
            project_root=project_root,
        )
        return gate, gate.snapshot_for((tool,))

    def _run_before_hook(
        self, gate: ToolGate, snapshot: ToolGateSnapshot, path: str
    ) -> BeforeToolCallResult | None:
        hook = gate.build_before_tool_hook(snapshot)
        return hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x",
                    name="write_file",
                    arguments={"path": path},
                ),
                args={"path": path},
                context=AgentContext(),
            ),
            None,
        )

    def test_parent_gate_relative_path_inside_workspace_allowed(self) -> None:
        """Parent gate with project_root: 工作区内相对路径通过。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        self.assertIsNone(result)

    def test_parent_gate_external_absolute_path_blocked(self) -> None:
        """绝对路径被 StructuredBoundaryPolicyEvaluator 阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "/etc/passwd")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_parent_gate_path_traversal_blocked(self) -> None:
        """../ 路径遍历被 _is_external_path 阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "../../etc/passwd")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_git_path_denied(self) -> None:
        """.git 路径被阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, ".git/config")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_sensitive_path_denied(self) -> None:
        """.env 文件路径被阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, ".env")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_restricted_dirs_still_blocks_with_project_root(self) -> None:
        """restricted_dirs 在 project_root 设置时仍独立生效（Tier 0）。"""
        mode = ExecutionModeState()
        tool = ToolSpec(
            "read_file", "Read.", "path", lambda v: "", schema=self.PATH_SCHEMA
        )
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            restricted_dirs=("secrets",),
            hook_manager=None,
            audit_logger=None,
            session_id="boundary-test",
            project_root=self._root,
        )
        snapshot = gate.snapshot_for((tool,))
        result = self._run_before_hook(gate, snapshot, "secrets/key.txt")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)

    def test_no_project_root_allows_relative_path_no_resolution(self) -> None:
        """project_root 未设置时，相对路径不被解析但仍通过（兼容旧行为）。"""
        gate, snapshot = self._make_file_gate(project_root=None)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        self.assertIsNone(result)

    def test_subagent_gate_uses_child_root(self) -> None:
        """子代理 gate 使用 child_root 进行边界解析，child_root 内路径通过。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        self.assertIsNone(result)

    def test_subagent_relative_path_inside_child_root_passes(self) -> None:
        """子代理 gate 使用 child_root，相对路径在 child_root 内可通过。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        self.assertIsNone(result)

    def test_subagent_gate_resolves_relative_path(self) -> None:
        """子代理 gate 使用 project_root 后，相对路径能被 StructuredBoundaryPolicyEvaluator 解析。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        inner = self._root / "inner"
        inner.mkdir(exist_ok=True)
        (self._root / "outer.txt").write_text("")
        result = self._run_before_hook(gate, snapshot, "inner/new.txt")
        self.assertIsNone(result)

    def test_subagent_no_approval_callback_ask_blocked(self) -> None:
        """子代理有 project_root 时，ask 仍因无 approval_callback 阻断。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy((PermissionRule("bash", "ask"),)),
            hook_manager=None,
            audit_logger=None,
            session_id="subagent-test",
            project_root=self._root,
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": "git push"}
                ),
                args={"command": "git push"},
                context=AgentContext(),
            ),
            None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.block)


if __name__ == "__main__":
    unittest.main()
