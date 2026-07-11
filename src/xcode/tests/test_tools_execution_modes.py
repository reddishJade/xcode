"""执行模式与策略单元测试。"""

from __future__ import annotations

from xcode.coding_agent.execution_modes import (
    parse_execution_mode,
    mode_notice,
    PlanPolicy,
    BuildPolicy,
    ActPolicy,
    ExecutionModeState,
)
from xcode.ai.events import ToolCall


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
