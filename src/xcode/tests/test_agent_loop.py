"""Agent 核心循环纯函数单元测试。"""

from __future__ import annotations

from xcode.agent.agent_loop import (
    _should_continue_max_tokens,
    _update_continuation_count,
    _has_empty_text_response,
    _assistant_error_detail,
    _cancelled_message,
    _continuation_limit_message,
)
from xcode.agent.config import AgentLoopConfig
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import TextContent


class TestShouldContinueMaxTokens:
    def test_max_tokens_with_continuation_enabled(self) -> None:
        config = AgentLoopConfig(max_tokens_continuation=True)
        assert _should_continue_max_tokens("max_tokens", config)

    def test_end_turn_no_continuation(self) -> None:
        config = AgentLoopConfig(max_tokens_continuation=True)
        assert not _should_continue_max_tokens("end_turn", config)

    def test_continuation_disabled(self) -> None:
        config = AgentLoopConfig(max_tokens_continuation=False)
        assert not _should_continue_max_tokens("max_tokens", config)


class TestUpdateContinuationCount:
    def test_short_output_increments(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="short")],
            stop_reason="max_tokens",
        )
        result = _update_continuation_count(
            msg, 0, AgentLoopConfig(min_continuation_tokens=500)
        )
        assert result == 1

    def test_long_output_resets(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="x" * 5000)],
            stop_reason="max_tokens",
        )
        result = _update_continuation_count(
            msg, 0, AgentLoopConfig(min_continuation_tokens=500)
        )
        assert result == 0

    def test_exceeds_limit_returns_none(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="short")],
            stop_reason="max_tokens",
        )
        result = _update_continuation_count(
            msg,
            2,
            AgentLoopConfig(
                min_continuation_tokens=500, max_consecutive_continuations=3
            ),
        )
        assert result is None


class TestHasEmptyTextResponse:
    def test_empty_content(self) -> None:
        msg = AssistantMessage(content=[], stop_reason="error")
        assert _has_empty_text_response(msg)

    def test_empty_text(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason="error",
        )
        assert _has_empty_text_response(msg)

    def test_non_empty_text(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="hello")],
            stop_reason="error",
        )
        assert not _has_empty_text_response(msg)


class TestAssistantErrorDetail:
    def test_from_error_message(self) -> None:
        msg = AssistantMessage(error_message="API error")
        assert _assistant_error_detail(msg) == "API error"

    def test_from_text_content(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="Something went wrong")],
        )
        assert _assistant_error_detail(msg) == "Something went wrong"

    def test_none(self) -> None:
        msg = AssistantMessage()
        assert _assistant_error_detail(msg) is None


class TestCancelledMessage:
    def test_no_signal(self) -> None:
        msg = _cancelled_message(None)
        assert msg.stop_reason == "aborted"
        assert "interrupted" in (msg.error_message or "")

    def test_with_signal(self) -> None:
        class MockSignal:
            reason = "user cancelled"

            def is_cancelled(self) -> bool:
                return True

        msg = _cancelled_message(MockSignal())
        assert msg.stop_reason == "aborted"
        assert "user cancelled" in (msg.error_message or "")


class TestContinuationLimitMessage:
    def test_message_content(self) -> None:
        config = AgentLoopConfig(min_continuation_tokens=500)
        msg = _continuation_limit_message(config)
        assert msg.stop_reason == "error"
        assert "500" in (msg.content[0].text if msg.content else "")
