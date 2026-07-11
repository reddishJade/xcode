"""Token 估算与压缩触发单元测试。"""

from __future__ import annotations

from xcode.agent._compaction import (
    estimate_tokens,
    estimate_message_tokens,
    should_compact_token_aware,
    extract_prompt_tokens_from_usage,
)
from xcode.agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ThinkingContent, ToolCallContent


class TestEstimateTokens:
    def test_non_empty_text(self) -> None:
        assert estimate_tokens("hello world") > 0

    def test_empty_text_returns_one(self) -> None:
        assert estimate_tokens("") == 1


class TestEstimateMessageTokens:
    def test_empty_list(self) -> None:
        assert estimate_message_tokens([]) == 0

    def test_system_message(self) -> None:
        msgs = [SystemMessage(content="hello")]
        assert estimate_message_tokens(msgs) > 0

    def test_assistant_thinking_content(self) -> None:
        msgs = [
            AssistantMessage(
                content=[ThinkingContent(thinking="deep thoughts")],
                stop_reason="end_turn",
            )
        ]
        assert estimate_message_tokens(msgs) > 0

    def test_assistant_tool_call(self) -> None:
        msgs = [
            AssistantMessage(
                content=[
                    ToolCallContent(id="c1", name="get", arguments={"key": "val"})
                ],
                stop_reason="tool_use",
            )
        ]
        assert estimate_message_tokens(msgs) > 0

    def test_mixed_messages(self) -> None:
        msgs = [
            UserMessage(content="hello"),
            AssistantMessage(
                content=[TextContent(text="world")], stop_reason="end_turn"
            ),
            ToolResultMessage(tool_call_id="c1", tool_name="t", content="result"),
        ]
        assert estimate_message_tokens(msgs) > 0


class TestShouldCompactTokenAware:
    def test_last_prompt_tokens_triggers(self) -> None:
        assert should_compact_token_aware([], last_prompt_tokens=32000)

    def test_last_prompt_tokens_below_threshold(self) -> None:
        assert not should_compact_token_aware([], last_prompt_tokens=100)

    def test_message_count_threshold(self) -> None:
        msgs = [UserMessage(content="x") for _ in range(100)]
        assert should_compact_token_aware(msgs, compact_threshold=50)

    def test_message_count_below_threshold(self) -> None:
        msgs = [UserMessage(content="x") for _ in range(10)]
        assert not should_compact_token_aware(msgs, compact_threshold=50)

    def test_no_trigger_default_false(self) -> None:
        assert not should_compact_token_aware([])

    def test_compact_token_threshold(self) -> None:
        msgs = [UserMessage(content="x" * 10000) for _ in range(10)]
        assert should_compact_token_aware(msgs, compact_token_threshold=100)


class TestExtractPromptTokensFromUsage:
    def test_none_usage(self) -> None:
        assert extract_prompt_tokens_from_usage(None) is None

    def test_empty_usage(self) -> None:
        assert extract_prompt_tokens_from_usage({}) is None

    def test_valid_prompt_tokens(self) -> None:
        assert extract_prompt_tokens_from_usage({"prompt_tokens": 150}) == 150

    def test_non_int_prompt_tokens(self) -> None:
        assert extract_prompt_tokens_from_usage({"prompt_tokens": "abc"}) is None
