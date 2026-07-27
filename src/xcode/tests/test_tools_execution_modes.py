"""执行模式与策略单元测试。"""

from __future__ import annotations

from pathlib import Path

from xcode.coding_agent.execution_modes import (
    DEFAULT_MODE_FALLBACKS,
    DEFAULT_SHELL_UNRESOLVED_POLICIES,
    build_default_mode_rulesets,
    parse_execution_mode,
    mode_notice,
    PlanPolicy,
    BuildPolicy,
    ActPolicy,
    ExecutionModeState,
)
from xcode.ai.events import ToolCall
from xcode.harness.agent_runtime.tool_gate import ToolGate


class TestParseExecutionMode:
    def test_plan(self) -> None:
        assert parse_execution_mode("plan") == "plan"

    def test_build(self) -> None:
        assert parse_execution_mode("build") == "build"

    def test_act(self) -> None:
        assert parse_execution_mode("act") == "act"

    def test_none_for_invalid(self) -> None:
        assert parse_execution_mode("unknown") is None

    def test_none_for_non_string(self) -> None:
        assert parse_execution_mode(123) is None


class TestModeNotice:
    def test_plan(self) -> None:
        notice = mode_notice("plan")
        assert "Plan Mode" in notice

    def test_build(self) -> None:
        notice = mode_notice("build")
        assert "Build Mode" in notice

    def test_act(self) -> None:
        notice = mode_notice("act")
        assert "Act Mode" in notice

    def test_unknown(self) -> None:
        assert mode_notice("unknown") == ""


class TestDefaultModeRulesets:
    def test_returns_rules_without_mutating_security_globals(
        self, tmp_path: Path
    ) -> None:
        rulesets = build_default_mode_rulesets(tmp_path)
        assert set(rulesets) == {"plan", "build", "act"}
        assert DEFAULT_MODE_FALLBACKS == {
            "plan": "deny",
            "build": "allow",
            "act": "ask",
        }
        assert DEFAULT_SHELL_UNRESOLVED_POLICIES == {
            "build": "allow",
            "act": "ask",
        }
        plan_patterns = {
            rule.resource_pattern
            for rule in rulesets["plan"]
            if rule.resource_pattern is not None
        }
        assert (tmp_path / ".xcode" / "plans" / "*.md").as_posix() in plan_patterns


class TestPlanPolicy:
    def test_filter_keeps_read_tools(self) -> None:
        from xcode.agent.types import ToolSpec

        tools = (
            ToolSpec(
                name="read_file", description="", input_hint="", handler=lambda d, _: ""
            ),
            ToolSpec(
                name="bash", description="", input_hint="", handler=lambda d, _: ""
            ),
        )
        filtered = PlanPolicy().filter_tools(tools)
        names = {t.name for t in filtered}
        assert "read_file" in names
        assert "bash" not in names


class TestBuildPolicy:
    def test_filter_keeps_all(self) -> None:
        from xcode.agent.types import ToolSpec

        tools = (
            ToolSpec(
                name="read_file", description="", input_hint="", handler=lambda d, _: ""
            ),
            ToolSpec(
                name="bash", description="", input_hint="", handler=lambda d, _: ""
            ),
        )
        assert len(BuildPolicy().filter_tools(tools)) == 2


class TestActPolicy:
    def test_filter_keeps_all(self) -> None:
        from xcode.agent.types import ToolSpec

        tools = (
            ToolSpec(
                name="read_file", description="", input_hint="", handler=lambda d, _: ""
            ),
            ToolSpec(
                name="bash", description="", input_hint="", handler=lambda d, _: ""
            ),
        )
        assert len(ActPolicy().filter_tools(tools)) == 2


class TestExecutionModeState:
    def test_default_is_act(self) -> None:
        state = ExecutionModeState()
        assert state.current_mode == "act"

    def test_set_mode(self) -> None:
        state = ExecutionModeState()
        state.set_mode("plan")
        assert state.current_mode == "plan"

    def test_plan_timeout_switches_to_build(self) -> None:
        state = ExecutionModeState(max_plan_turns=3)
        state.set_mode("plan")
        for _ in range(2):
            assert not state.check_plan_timeout()
        assert state.check_plan_timeout()
        assert state.current_mode == "build"

    def test_non_plan_timeout_noop(self) -> None:
        state = ExecutionModeState()
        assert not state.check_plan_timeout()
        assert state.current_mode == "act"

    def test_check_call_delegates_to_policy(self) -> None:
        state = ExecutionModeState()
        call = ToolCall(id="c1", name="read_file", input={})
        result = state.check_call(call)
        assert result in ("allow", "deny", "ask")

    def test_tool_gate_freezes_shell_policy_for_current_mode(self) -> None:
        state = ExecutionModeState()
        gate = ToolGate(
            mode_state=state,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="test",
            shell_unresolved_policies=DEFAULT_SHELL_UNRESOLVED_POLICIES,
        )

        state.set_mode("build")
        build_snapshot = gate.snapshot()
        state.set_mode("act")
        act_snapshot = gate.snapshot()

        assert build_snapshot.shell_unresolved_policy == "allow"
        assert act_snapshot.shell_unresolved_policy == "ask"
