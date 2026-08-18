"""Agent 循环结果类型单元测试。"""

from __future__ import annotations

from xcode.agent.messages import UserMessage
from xcode.agent.results import AgentLoopResult, TerminationReason


class TestTerminationReason:
    def test_values(self) -> None:
        assert TerminationReason.COMPLETED == "completed"
        assert TerminationReason.CANCELLED == "cancelled"
        assert TerminationReason.STEP_LIMIT == "step_limit"
        assert TerminationReason.WATCHDOG == "watchdog"
        assert TerminationReason.PROVIDER_ERROR == "provider_error"


class TestAgentLoopResult:
    def test_surface_is_explicit(self) -> None:
        result = AgentLoopResult(
            messages=[],
            surface=[UserMessage(content="current")],
        )
        assert result.steps == 0
        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.surface == [UserMessage(content="current")]
