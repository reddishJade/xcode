"""Agent 循环结果类型单元测试。"""

from __future__ import annotations

from xcode.agent.results import AgentLoopResult, TerminationReason


class TestTerminationReason:
    def test_values(self) -> None:
        assert TerminationReason.COMPLETED == "completed"
        assert TerminationReason.CANCELLED == "cancelled"
        assert TerminationReason.STEP_LIMIT == "step_limit"
        assert TerminationReason.WATCHDOG == "watchdog"
        assert TerminationReason.PROVIDER_ERROR == "provider_error"


class TestAgentLoopResult:
    def test_defaults(self) -> None:
        result = AgentLoopResult()
        assert result.steps == 0
        assert result.termination_reason == TerminationReason.COMPLETED

    def test_stopped_by_limit(self) -> None:
        result = AgentLoopResult(termination_reason=TerminationReason.STEP_LIMIT)
        assert result.stopped_by_limit

    def test_stopped_by_watchdog(self) -> None:
        result = AgentLoopResult(termination_reason=TerminationReason.WATCHDOG)
        assert result.stopped_by_watchdog

    def test_stopped_by_error(self) -> None:
        result = AgentLoopResult(termination_reason=TerminationReason.PROVIDER_ERROR)
        assert result.stopped_by_error

    def test_completed_not_stopped(self) -> None:
        result = AgentLoopResult(termination_reason=TerminationReason.COMPLETED)
        assert not result.stopped_by_limit
        assert not result.stopped_by_watchdog
        assert not result.stopped_by_error
