from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
import tempfile
from xcode.harness.observability import (
    HITLResult,
    HookManager,
    InMemoryGrantStore,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
    ActionExtractor,
    create_grant_record,
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
import pytest

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


class XcodePermissionsTests:
    def test_permission_policy_denies_tool(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda value: value["input"])
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="echo", decision="deny"),)
                ),
            )
        )
        result = engine.decide("echo", {"input": "hello"}, tool_spec=tool)
        assert result.blocked
        assert "deny for echo" in result.reason

    def test_handler_exception_reports_error(self) -> None:
        def fail(_value: dict) -> str:
            raise RuntimeError("boom")

        tool = ToolSpec("fail", "Fail.", "text", fail)
        result = PermissionEngine(PermissionEngineConfig()).decide(
            "fail", {}, tool_spec=tool
        )
        assert not (result.blocked)
        with pytest.raises(RuntimeError):
            tool.handler({})

    def test_permission_policy_ask_for_low_risk_tool(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda value: value["input"])
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="echo", decision="ask"),)
                ),
            )
        )
        result = engine.decide("echo", {"input": "hello"}, tool_spec=tool)
        assert result.blocked
        assert "requires approval" in result.reason

    def test_permission_policy_allow_skips_high_risk_approval(self) -> None:
        tool = ToolSpec(
            "danger",
            "Danger.",
            "text",
            lambda value: value["input"],
        )
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="danger", decision="allow"),)
                ),
            )
        )
        result = engine.decide("danger", {"input": "go"}, tool_spec=tool)
        assert not (result.blocked)
        assert tool.handler({"input": "go"}) == "go"

    def test_static_last_match_wins(self) -> None:
        rules = (
            StaticPermission(tool="bash", decision="allow"),
            StaticPermission(tool="*", decision="deny"),
        )
        policy = PermissionPolicy(rules)
        engine = PermissionEngine(PermissionEngineConfig(static_policy=policy))
        result = engine.decide("bash", {})
        assert result.blocked
        assert result.decision == "deny"

    def test_permission_engine_restricted_dirs_deny(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                restricted_dirs=("secrets",),
            )
        )
        result = engine.decide(
            "read_file",
            {"path": "secrets/key.txt"},
        )
        assert result.blocked
        assert result.matched_rule == "restricted_dirs"

    def test_restricted_dirs_does_not_match_plain_text(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(restricted_dirs=("secrets",)))
        result = engine.decide(
            "echo",
            {"input": "the secrets directory is documented"},
        )
        assert not (result.blocked)

    def test_restricted_dirs_rejects_prefix_collision_only_when_contained(
        self,
    ) -> None:
        engine = PermissionEngine(PermissionEngineConfig(restricted_dirs=("secrets",)))
        result = engine.decide(
            "read_file",
            {"path": "secrets-copy/key.txt"},
        )
        assert not (result.blocked)

    def test_restricted_dirs_checks_all_patch_targets(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(restricted_dirs=("secrets",)))
        result = engine.decide(
            "apply_patch",
            {"paths": ["src/app.py", "secrets/key.txt"]},
        )
        assert result.blocked
        assert result.matched_rule == "restricted_dirs"

    def test_restricted_dirs_asks_for_unparseable_filesystem_command(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(restricted_dirs=("secrets",)))
        result = engine.decide(
            "bash",
            {"command": 'rm "unterminated'},
        )
        assert result.blocked
        assert result.decision == "ask"

    def test_restricted_dirs_resolves_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            restricted = root / "secrets"
            restricted.mkdir()
            engine = PermissionEngine(
                PermissionEngineConfig(
                    restricted_dirs=(str(restricted),),
                    project_root=root,
                )
            )
            result = engine.decide(
                "read_file",
                {"path": str(restricted / "key.txt")},
            )
        assert result.blocked

    def test_restricted_dirs_resolves_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            restricted = root / "secrets"
            restricted.mkdir()
            link = root / "public"
            try:
                link.symlink_to(restricted, target_is_directory=True)
            except OSError:
                pytest.skip("Cannot create directory symlink in this environment")
            engine = PermissionEngine(
                PermissionEngineConfig(
                    restricted_dirs=("secrets",),
                    project_root=root,
                )
            )
            result = engine.decide(
                "read_file",
                {"path": "public/key.txt"},
            )
        assert result.blocked

    def test_restricted_dirs_checks_shell_path_target(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(restricted_dirs=("secrets",)))
        result = engine.decide(
            "bash",
            {"command": "cat secrets/key.txt"},
        )
        assert result.blocked
        assert result.decision == "deny"

    def test_permission_engine_session_grant_satisfies_ask(self) -> None:
        from xcode.harness.observability import (
            InMemoryGrantStore,
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
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="bash", decision="ask"),)
                ),
                session_grant_store=store,
            )
        )
        result = engine.decide(
            "bash",
            {"command": "git status"},
        )
        assert not (result.blocked)
        assert result.matched_rule == "session_grant"

    def test_global_default_ask_applies_when_no_rule_matches(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    rules=(StaticPermission(tool="read_file", decision="allow"),),
                    global_default="ask",
                ),
            )
        )
        allowed = engine.decide("read_file", {"path": "a.txt"})
        assert not (allowed.blocked)
        unknown = engine.decide("write_file", {"path": "b.txt"})
        assert unknown.blocked
        assert unknown.decision == "ask"

    def test_permission_engine_default_allow(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("any_tool", {"input": "anything"})
        assert not (result.blocked)
        assert result.matched_rule == "default"

    def test_permission_engine_high_risk_approval(self) -> None:
        # High-risk approval path removed (STEP 5). Default allow.
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("danger", {"input": "hello"})
        assert not (result.blocked)

    def test_permission_engine_execution_mode_deny(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide(
            "bash",
            {"command": "git status"},
            execution_decision="deny",
        )
        assert result.blocked
        assert result.matched_rule == "mode"

    def test_tool_gate_static_deny_preempts_execution_mode(self) -> None:
        called = False

        def approve(_tool: ToolSpec, _input: dict[str, object]) -> HITLResult:
            nonlocal called
            called = True
            return HITLResult("allow", "once")

        mode = ExecutionModeState()
        mode.set_mode("act")
        tool = ToolSpec("bash", "Bash.", "command", lambda _value: "")
        gate = ToolGate(
            mode_state=mode,
            approval_callback=approve,
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="deny"),)
            ),
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
        assert result is not None
        assert result is not None
        assert "deny for bash" in result.reason
        assert not (called)

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
                permission_policy=PermissionPolicy(
                    (StaticPermission(tool="echo", decision="deny"),)
                ),
            ),
        )

        result = agent.run("go")

        assert "deny for echo" in result.messages[2]["content"][0]["content"]

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

        assert result.messages[2]["content"][0]["content"] == "go"

    def test_permission_allows_then_handler_raises(self) -> None:
        def fail_handler(_value: dict) -> str:
            raise RuntimeError("handler failed")

        tool = ToolSpec("failing", "Fails.", "text", fail_handler)
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="failing", decision="ask"),)
                ),
            )
        )
        result = engine.decide(
            "failing",
            {"input": "go"},
            tool_spec=tool,
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
        )
        assert not (result.blocked)
        with pytest.raises(RuntimeError):
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
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="bash", decision="ask"),)
                ),
            )
        )
        result = engine.decide(
            "bash",
            {"command": "git add ."},
            tool_spec=tool,
            approval_callback=lambda _t, _i: HITLResult("deny", "once"),
        )
        assert result.blocked
        assert result.decision == "deny"

    def test_denied_callback_metadata(self) -> None:
        tool = ToolSpec(
            "bash",
            "Bash.",
            "text",
            lambda v: v["command"],
        )
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="bash", decision="ask"),)
                ),
            )
        )
        result = engine.decide(
            "bash",
            {"command": "git push"},
            tool_spec=tool,
            approval_callback=lambda _t, _i: HITLResult("deny", "session"),
        )
        assert result.blocked
        assert result.decision == "deny"


# ── Subagent Permission Boundary Tests ──


class SubagentGatePermissionBoundaryTests:
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
        assert result is not None
        assert result is not None
        assert result.block

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
        assert result is not None
        assert result is not None
        assert result.block

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
        assert result is None

    def test_subagent_static_deny_inherited(self) -> None:
        """子代理继承静态 deny 规则。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="deny"),)
            ),
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
        assert result is not None
        assert result is not None
        assert result.block

    def test_subagent_global_default_ask(self) -> None:
        """子代理使用 global_default="ask"，仅显式 allow 的工具可通过。"""
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
            permission_policy=PermissionPolicy(
                rules=(StaticPermission(tool="read_file", decision="allow"),),
                global_default="ask",
            ),
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
        assert allowed is None

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
        assert blocked is not None
        assert blocked is not None
        assert blocked.block

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
        assert result is not None
        assert result is not None
        assert result.block

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
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
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
        assert result is not None
        assert result is not None
        assert result.block
        assert len(called) == 0

    def test_subagent_no_session_grant_store(self) -> None:
        """子代理无 session grant store，ask 不能被已存在的 grant 满足。"""
        from xcode.harness.observability import (
            InMemoryGrantStore,
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
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
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
        assert result is not None
        assert result is not None
        assert result.block

    def test_subagent_permanent_grant_ignored(self) -> None:
        """子代理无 permanent grant store，永久授权不生效。"""
        from xcode.harness.observability import (
            FileGrantStore,
        )

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
                permission_policy=PermissionPolicy(
                    (StaticPermission(tool="bash", decision="ask"),)
                ),
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
            assert result is not None
            assert result is not None
            assert result.block
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
                permission_policy=PermissionPolicy(
                    (StaticPermission(tool="echo", decision="deny"),)
                ),
            ),
        )
        result = agent.run("go")
        assert "deny for echo" in result.messages[2]["content"][0]["content"]

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
        assert len(audit_records) > 0

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
        assert len(hook_records) > 0


# ── Project Root / Boundary Resolution Tests ──


class ToolGateBoundaryResolutionTests:
    """验证 project_root 传送到 StructuredBoundaryPolicyEvaluator 后
    能正确解析工作区相对路径。"""

    PATH_SCHEMA = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def setup_method(self, method) -> None:
        self._tmp = tempfile.mkdtemp()
        self._root = Path(self._tmp).resolve()
        (self._root / "subdir").mkdir()
        (self._root / "secrets").mkdir()

    def teardown_method(self, method) -> None:
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
        assert result is None

    def test_parent_gate_external_absolute_path_blocked(self) -> None:
        """绝对路径被 StructuredBoundaryPolicyEvaluator 阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "/etc/passwd")
        assert result is not None
        assert result is not None
        assert result.block

    def test_parent_gate_path_traversal_blocked(self) -> None:
        """../ 路径遍历被 _is_external_path 阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "../../etc/passwd")
        assert result is not None
        assert result is not None
        assert result.block

    def test_git_path_denied(self) -> None:
        """.git 路径被阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, ".git/config")
        assert result is not None
        assert result is not None
        assert result.block

    def test_sensitive_path_denied(self) -> None:
        """.env 文件路径被阻断。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, ".env")
        assert result is not None
        assert result is not None
        assert result.block

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
        assert result is not None
        assert result is not None
        assert result.block

    def test_no_project_root_allows_relative_path_no_resolution(self) -> None:
        """project_root 未设置时，相对路径不被解析但仍通过（兼容旧行为）。"""
        gate, snapshot = self._make_file_gate(project_root=None)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        assert result is None

    def test_subagent_gate_uses_child_root(self) -> None:
        """子代理 gate 使用 child_root 进行边界解析，child_root 内路径通过。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        assert result is None

    def test_subagent_relative_path_inside_child_root_passes(self) -> None:
        """子代理 gate 使用 child_root，相对路径在 child_root 内可通过。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        result = self._run_before_hook(gate, snapshot, "subdir/file.txt")
        assert result is None

    def test_subagent_gate_resolves_relative_path(self) -> None:
        """子代理 gate 使用 project_root 后，相对路径能被 StructuredBoundaryPolicyEvaluator 解析。"""
        gate, snapshot = self._make_file_gate(project_root=self._root)
        inner = self._root / "inner"
        inner.mkdir(exist_ok=True)
        (self._root / "outer.txt").write_text("")
        result = self._run_before_hook(gate, snapshot, "inner/new.txt")
        assert result is None

    def test_subagent_no_approval_callback_ask_blocked(self) -> None:
        """子代理有 project_root 时，ask 仍因无 approval_callback 阻断。"""
        mode = ExecutionModeState()
        tool = ToolSpec("bash", "Bash.", "command", lambda v: "", schema=INPUT_SCHEMA)
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
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
        assert result is not None
        assert result is not None
        assert result.block


class ToolGateGrantFlowTests:
    """验证 ToolGate 生产路径中的 canonical ask/grant/callback 流程。"""

    INPUT_SCHEMA = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    }

    def _make_gate(
        self,
        *,
        policy: PermissionPolicy | None = None,
        session_store: InMemoryGrantStore | None = None,
        callback: Callable | None = None,
    ) -> ToolGate:
        mode = ExecutionModeState()
        return ToolGate(
            mode_state=mode,
            approval_callback=callback,
            permission_policy=policy,
            hook_manager=None,
            audit_logger=None,
            session_id="toolgate-test",
            session_grant_store=session_store,
        )

    def _make_hook(self, gate: ToolGate) -> Callable:
        tool = ToolSpec(
            "bash", "Bash.", "command", lambda v: "", schema=self.INPUT_SCHEMA
        )
        return gate.build_before_tool_hook(gate.snapshot_for((tool,)))

    def _call_hook(
        self, hook, command: str = "echo hello"
    ) -> BeforeToolCallResult | None:
        return hook(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[]),
                tool_call=ToolCallContent(
                    id="x", name="bash", arguments={"command": command}
                ),
                args={"command": command},
                context=AgentContext(),
            ),
            None,
        )

    # ── 1. ask + no grant → callback called ──

    def test_ask_no_grant_calls_callback(self) -> None:
        """Static policy ask + no matching grant → callback invoked."""
        calls: list[bool] = []

        def cb(_tool, _input):
            calls.append(True)
            return HITLResult("allow", "session")

        gate = self._make_gate(
            policy=PermissionPolicy((StaticPermission(tool="bash", decision="ask"),)),
            session_store=InMemoryGrantStore(),
            callback=cb,
        )
        hook = self._make_hook(gate)
        result = self._call_hook(hook)
        assert result is None  # not blocked
        assert len(calls) == 1

    # ── 2. callback allow/session writes grant ──

    def test_callback_allow_session_writes_grant(self) -> None:
        """Callback returning allow/session writes a grant to session store."""
        store = InMemoryGrantStore()
        gate = self._make_gate(
            policy=PermissionPolicy((StaticPermission(tool="bash", decision="ask"),)),
            session_store=store,
            callback=lambda _t, _i: HITLResult("allow", "session"),
        )
        hook = self._make_hook(gate)
        self._call_hook(hook, "git status")
        assert len(store.records()) == 1
        assert store.records()[0].decision == "allow"
        assert store.records()[0].scope == "session"

    # ── 3. same session reuses grant, callback not called again ──

    def test_same_session_reuses_grant(self) -> None:
        """Second call with same session store reuses grant, no callback."""
        callback_calls: list[int] = []

        def cb(_t, _i):
            callback_calls.append(1)
            return HITLResult("allow", "session")

        store = InMemoryGrantStore()
        gate = self._make_gate(
            policy=PermissionPolicy((StaticPermission(tool="bash", decision="ask"),)),
            session_store=store,
            callback=cb,
        )
        hook = self._make_hook(gate)

        # First call: no grant → callback → grant written
        self._call_hook(hook, "git status")
        assert len(callback_calls) == 1

        # Second call: grant reused → callback NOT called
        self._call_hook(hook, "git status")
        assert len(callback_calls) == 1

    # ── 4. different session store does not reuse grant ──

    def test_different_store_does_not_reuse_grant(self) -> None:
        """Switching to a different session store does not reuse old grant."""
        callback_calls: list[int] = []

        def cb(_t, _i):
            callback_calls.append(1)
            return HITLResult("allow", "session")

        store_a = InMemoryGrantStore()
        store_b = InMemoryGrantStore()
        mode = ExecutionModeState()
        tool = ToolSpec(
            "bash", "Bash.", "command", lambda v: "", schema=self.INPUT_SCHEMA
        )

        # Session A: first call writes grant
        gate = ToolGate(
            mode_state=mode,
            approval_callback=cb,
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
            hook_manager=None,
            audit_logger=None,
            session_id="test-a",
            session_grant_store=store_a,
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        self._call_hook(hook, "git status")
        assert len(store_a.records()) == 1
        assert len(callback_calls) == 1

        # Session B: different store, grant must be re-approved
        gate_b = ToolGate(
            mode_state=mode,
            approval_callback=cb,
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
            hook_manager=None,
            audit_logger=None,
            session_id="test-b",
            session_grant_store=store_b,
        )
        hook_b = gate_b.build_before_tool_hook(gate_b.snapshot_for((tool,)))
        self._call_hook(hook_b, "git status")
        assert len(store_b.records()) == 1  # new grant written
        assert len(callback_calls) == 2  # callback called again

    # ── 5. callback deny blocks the tool ──

    def test_callback_deny_blocks_tool(self) -> None:
        """Callback returning deny blocks the tool."""
        gate = self._make_gate(
            policy=PermissionPolicy((StaticPermission(tool="bash", decision="ask"),)),
            session_store=InMemoryGrantStore(),
            callback=lambda _t, _i: HITLResult("deny", "once"),
        )
        hook = self._make_hook(gate)
        result = self._call_hook(hook, "rm -rf /")
        assert result is not None
        assert result is not None
        assert result.block

    # ── 6. no store + no callback → ask blocks ──

    def test_ask_blocks_when_no_approval_mechanism(self) -> None:
        """No grant store and no approval_callback means ask blocks."""
        mode = ExecutionModeState()
        tool = ToolSpec(
            "bash", "Bash.", "command", lambda v: "", schema=self.INPUT_SCHEMA
        )
        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
            hook_manager=None,
            audit_logger=None,
            session_id="test-no-mechanism",
            # no session_grant_store, no permanent_grant_store
        )
        hook = gate.build_before_tool_hook(gate.snapshot_for((tool,)))
        result = self._call_hook(hook, "echo hello")
        assert result is not None
        assert result is not None
        assert result.block


if __name__ == "__main__":
    pytest.main()
