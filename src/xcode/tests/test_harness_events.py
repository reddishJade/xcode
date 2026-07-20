"""AgentEvent 到 AgentHarnessEvent 的翻译纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.events import (
    _translate_event,
    _StreamTranslationState,
    _translate_message_update,
    _translate_turn_end,
    _translate_thinking_update,
    _translate_tool_execution_start,
    _translate_tool_execution_end,
    _translate_compaction,
    _tool_update_text,
    TextDeltaStructuredEvent,
    ReasoningDeltaStructuredEvent,
    ToolUseStructuredEvent,
    ToolResultStructuredEvent,
    CompactionStructuredEvent,
    TurnEndStructuredEvent,
)
from xcode.agent.events import (
    AgentStartEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageUpdateEvent,
    ThinkingUpdateEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    CompactionEvent,
)
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import TextContent, ToolCallContent, AgentToolResult


def test_translate_start_event_returns_none() -> None:
    state = _StreamTranslationState()
    assert _translate_event(AgentStartEvent(), state) is None


def test_turn_start_increments_step() -> None:
    state = _StreamTranslationState()
    assert state.step == 0
    _translate_event(TurnStartEvent(), state)
    assert state.step == 1


def test_thinking_update() -> None:
    state = _StreamTranslationState()
    result = _translate_thinking_update(
        ThinkingUpdateEvent(reasoning_content="thinking..."), state
    )
    assert isinstance(result, ReasoningDeltaStructuredEvent)
    assert result.data == "thinking..."


def test_tool_execution_start() -> None:
    state = _StreamTranslationState()
    result = _translate_tool_execution_start(
        ToolExecutionStartEvent(
            tool_call_id="c1", tool_name="read_file", args={"path": "/x"}
        ),
        state,
    )
    assert isinstance(result, ToolUseStructuredEvent)
    assert result.data.name == "read_file"


def test_tool_execution_end() -> None:
    from xcode.agent.messages import ToolResultMessage

    state = _StreamTranslationState()
    result = _translate_tool_execution_end(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="read_file",
            result=ToolResultMessage(
                tool_call_id="c1", tool_name="read_file", content="result text"
            ),
            is_error=False,
        ),
        state,
    )
    assert isinstance(result, ToolResultStructuredEvent)
    assert result.data.status == "ok"


def test_tool_execution_end_error() -> None:
    state = _StreamTranslationState()
    result = _translate_tool_execution_end(
        ToolExecutionEndEvent(
            tool_call_id="c1", tool_name="bash", result=None, is_error=True
        ),
        state,
    )
    assert isinstance(result, ToolResultStructuredEvent)
    assert result.data.status == "error"


def test_compaction_event() -> None:
    state = _StreamTranslationState()
    result = _translate_compaction(
        CompactionEvent(
            messages_removed=5,
            messages_after=3,
            summary_token_estimate=200,
            trigger="token_limit",
        ),
        state,
    )
    assert isinstance(result, CompactionStructuredEvent)
    assert result.data.messages_removed == 5


class TestTranslateMessageUpdate:
    def test_text_delta_extraction(self) -> None:
        state = _StreamTranslationState()
        state.step = 1
        msg = AssistantMessage(content=[TextContent(text="Hello")])
        result = _translate_message_update(MessageUpdateEvent(message=msg), state)
        assert isinstance(result, TextDeltaStructuredEvent)
        assert result.data == "Hello"

    def test_delta_subsequent(self) -> None:
        state = _StreamTranslationState()
        state.step = 1
        state.text_seen[1] = "Hel"
        msg = AssistantMessage(content=[TextContent(text="Hello")])
        result = _translate_message_update(MessageUpdateEvent(message=msg), state)
        assert result.data == "lo"

    def test_non_assistant_returns_none(self) -> None:
        state = _StreamTranslationState()
        result = _translate_message_update(MessageUpdateEvent(message=None), state)
        assert result is None


class TestTranslateTurnEnd:
    def test_tool_results_included(self) -> None:
        state = _StreamTranslationState()
        from xcode.agent.messages import ToolResultMessage

        event = TurnEndEvent(
            message=AssistantMessage(content=[TextContent(text="done")]),
            tool_results=[
                ToolResultMessage(
                    tool_call_id="c1", tool_name="read", content="output"
                ),
            ],
        )
        result = _translate_turn_end(event, state)
        assert isinstance(result, TurnEndStructuredEvent)
        assert len(result.data.tool_results) == 1


class TestToolUpdateText:
    def test_text_content(self) -> None:
        result = AgentToolResult(content=[TextContent(text="hello")])
        assert _tool_update_text(result) == "hello"

    def test_mixed_content(self) -> None:
        result = AgentToolResult(
            content=[TextContent(text="a"), ToolCallContent(id="c1", name="search")]
        )
        assert _tool_update_text(result) == "a"

    def test_none(self) -> None:
        assert _tool_update_text(None) == ""
