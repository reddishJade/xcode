"""Agent 消息到 LLM 格式转换单元测试。"""

from __future__ import annotations

from xcode.agent._codec import convert_to_llm
from xcode.agent.messages import (
    AssistantMessage,
    CompactionSummaryMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ToolCallContent, ToolResultContent


class TestConvertSystemMessage:
    def test_system_message(self) -> None:
        result = convert_to_llm([SystemMessage(content="You are helpful.")])
        assert result == [{"role": "system", "content": "You are helpful."}]


class TestConvertUserMessage:
    def test_str_content(self) -> None:
        result = convert_to_llm([UserMessage(content="hello")])
        assert result == [{"role": "user", "content": "hello"}]


class TestConvertAssistantMessage:
    def test_text_content(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="Hello!")],
            stop_reason="end_turn",
        )
        result = convert_to_llm([msg])
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello!"

    def test_tool_calls(self) -> None:
        msg = AssistantMessage(
            content=[
                TextContent(text="Let me search."),
                ToolCallContent(id="c1", name="search", arguments={"q": "x"}),
            ],
            stop_reason="tool_use",
        )
        result = convert_to_llm([msg])
        assert result[0]["content"] == "Let me search."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_reasoning_content_passed_through(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="Answer")],
            reasoning_content="I think...",
            stop_reason="end_turn",
        )
        result = convert_to_llm([msg])
        assert result[0]["reasoning_content"] == "I think..."


class TestConvertToolResultMessage:
    def test_str_content(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c1",
            tool_name="search",
            content='{"result": "ok"}',
            is_error=False,
        )
        result = convert_to_llm([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "c1"
        assert result[0]["content"] == '{"result": "ok"}'

    def test_list_content_flattened(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c1",
            tool_name="bash",
            content=[
                TextContent(text="line1"),
                TextContent(text="line2"),
                ToolResultContent(tool_use_id="", content="block"),
            ],
            is_error=False,
        )
        result = convert_to_llm([msg])
        assert result[0]["role"] == "tool"
        assert "line1" in result[0]["content"]
        assert "line2" in result[0]["content"]
        assert "block" in result[0]["content"]


class TestConvertCompactionSummary:
    def test_compaction_summary_wraps_in_tags(self) -> None:
        msg = CompactionSummaryMessage(
            summary="Previous context was...", tokens_before=500
        )
        result = convert_to_llm([msg])
        assert result[0]["role"] == "user"
        text = str(result[0]["content"])
        assert "<summary>" in text
        assert "Previous context was..." in text
        assert "</summary>" in text
