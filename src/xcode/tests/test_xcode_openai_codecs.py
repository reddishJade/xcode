from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from xcode.agent.message_converter import convert_to_llm
from xcode.agent.messages import (
    AssistantMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    ToolResultMessage,
)
from xcode.agent.types import FileContent, ImageContent, TextContent, ToolCallContent
from xcode.ai.events import ReasoningDelta, TextDelta, ToolCallEvent
from xcode.ai.providers.codec import to_chat_messages, to_chat_tool
from xcode.ai.providers.stream_codec import chat_stream_to_events
from xcode.harness.skills import ToolSpec
import pytest


class OpenAIToolCodecTest:
    def test_tool_schema_uses_explicit_schema(self) -> None:
        tool = ToolSpec(
            "echo",
            "Echo input.",
            'JSON: {"text":"..."}',
            lambda _value: "",
            schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        encoded = to_chat_tool(tool.name, tool.description, tool.schema)

        assert encoded["function"]["parameters"] == tool.schema

    def test_tool_results_convert_to_openai_tool_messages(self) -> None:
        messages = to_chat_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "echo",
                            "input": {"text": "hi"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "hi"}
                    ],
                },
            ]
        )

        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "tool"

    def test_tool_call_arguments_are_serialized_for_chat_api(self) -> None:
        messages = to_chat_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {
                                "name": "grep_search",
                                "arguments": {"path": "src/xcode"},
                            },
                        }
                    ],
                }
            ]
        )

        arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(arguments, str)
        assert arguments == '{"path":"src/xcode"}'

    def test_reasoning_content_is_preserved_for_thinking_mode(self) -> None:
        messages = to_chat_messages(
            [
                {
                    "role": "assistant",
                    "reasoning_content": "private reasoning",
                    "content": [
                        {"type": "text", "text": "I will search."},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "grep_search",
                            "input": {"input": "skill"},
                        },
                    ],
                }
            ]
        )

        assert messages[0]["reasoning_content"] == "private reasoning"

    def test_empty_reasoning_content_is_preserved_for_tool_calls(self) -> None:
        raw_messages = convert_to_llm(
            [
                AssistantMessage(
                    content=[
                        ToolCallContent(
                            id="t1",
                            name="grep_search",
                            arguments={"input": "skill"},
                        )
                    ],
                    reasoning_content="",
                )
            ]
        )

        messages = to_chat_messages(raw_messages)

        assert messages[0]["reasoning_content"] == ""

    def test_agent_message_discriminators_are_pythonic_inside_boundary(self) -> None:
        tool_call = ToolCallContent(id="t1", name="grep_search")
        assert tool_call.type == "tool_call"
        assert ToolResultMessage().role == "tool_result"
        assert BranchSummaryMessage().role == "branch_summary"
        assert CompactionSummaryMessage().role == "compaction_summary"

        raw_messages = convert_to_llm(
            [
                AssistantMessage(content=[tool_call]),
                ToolResultMessage(tool_call_id="t1", content="done"),
            ]
        )

        assert raw_messages[0]["tool_calls"][0]["id"] == "t1"
        assert raw_messages[1]["role"] == "tool"

    def test_typed_tool_results_do_not_inline_binary_data(self) -> None:
        """provider 文本只保留类型摘要，不复制图片或文件数据。"""
        image = ImageContent(
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "secret-image-data",
            }
        )
        file = FileContent(
            filename="audio.wav",
            file_data="secret-audio-data",
        )
        raw_messages = convert_to_llm(
            [
                ToolResultMessage(
                    tool_call_id="t1",
                    content=[
                        TextContent(text="summary"),
                        image,
                        file,
                    ],
                )
            ]
        )

        content = raw_messages[0]["content"]
        assert content == "summary[image result: image/png][file result: audio.wav]"
        assert "secret-image-data" not in content
        assert "secret-audio-data" not in content
        assert "secret-image-data" not in repr(image)
        assert "secret-audio-data" not in repr(file)


class OpenAIStreamCodecTest:
    def test_chat_stream_aggregates_tool_call_arguments(self) -> None:
        events = list(
            chat_stream_to_events(
                [
                    FakeStreamChunk(content="he"),
                    FakeStreamChunk(content="llo"),
                    FakeStreamChunk(
                        FakeStreamToolCall(
                            0, call_id="call-1", name="echo", arguments='{"text": '
                        )
                    ),
                    FakeStreamChunk(FakeStreamToolCall(0, arguments='"hi"}')),
                ]
            )
        )

        assert isinstance(events[0], TextDelta)
        first_text = cast(TextDelta, events[0])
        assert first_text.chunk == "he"
        assert isinstance(events[-1], ToolCallEvent)
        final_call = cast(ToolCallEvent, events[-1])
        assert final_call.calls[0].input == {"text": "hi"}

    def test_chat_stream_extracts_reasoning_content(self) -> None:
        events = list(
            chat_stream_to_events(
                [
                    FakeStreamChunk(reasoning_content="I am thinking"),
                    FakeStreamChunk(reasoning_content=" deeply"),
                    FakeStreamChunk(content="Hello"),
                ]
            )
        )
        assert len(events) == 3
        assert isinstance(events[0], ReasoningDelta)
        first_reasoning = cast(ReasoningDelta, events[0])
        assert first_reasoning.chunk == "I am thinking"
        assert isinstance(events[1], ReasoningDelta)
        second_reasoning = cast(ReasoningDelta, events[1])
        assert second_reasoning.chunk == " deeply"
        assert isinstance(events[2], TextDelta)
        final_text = cast(TextDelta, events[2])
        assert final_text.chunk == "Hello"

    def test_chat_stream_handles_chunks_without_usage_attribute(self) -> None:
        events = list(
            chat_stream_to_events(
                cast(
                    Any,
                    [
                        FakeStreamChunkNoUsage(content="hello"),
                    ],
                )
            )
        )

        assert len(events) == 1
        assert isinstance(events[0], TextDelta)
        assert cast(TextDelta, events[0]).chunk == "hello"


class FakeStreamChunk:
    def __init__(
        self,
        tool_call: FakeStreamToolCall | None = None,
        content: str | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self._choices = [FakeStreamChoice(content, tool_call, reasoning_content)]
        self._usage: None = None

    @property
    def choices(self) -> Sequence[FakeStreamChoice]:
        return self._choices

    @property
    def usage(self) -> None:
        return self._usage


class FakeStreamChunkNoUsage:
    def __init__(self, content: str | None = None) -> None:
        self.choices = [FakeStreamChoice(content, None, None)]


class FakeStreamChoice:
    def __init__(
        self,
        content: str | None,
        tool_call: FakeStreamToolCall | None,
        reasoning_content: str | None = None,
    ) -> None:
        self._delta = FakeStreamDelta(content, tool_call, reasoning_content)

    @property
    def delta(self) -> FakeStreamDelta:
        return self._delta


class FakeStreamDelta:
    def __init__(
        self,
        content: str | None,
        tool_call: FakeStreamToolCall | None,
        reasoning_content: str | None = None,
    ) -> None:
        self._content: str | None = content
        self._tool_calls: list[FakeStreamToolCall] | None = (
            [tool_call] if tool_call is not None else []
        )
        self.reasoning_content: str | None = reasoning_content

    @property
    def content(self) -> str | None:
        return self._content

    @property
    def tool_calls(self) -> Sequence[FakeStreamToolCall] | None:
        return self._tool_calls


class FakeStreamToolCall:
    def __init__(
        self,
        index: int,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        self._index: int = index
        self._id: str | None = call_id
        self._function: FakeStreamFunction | None = FakeStreamFunction(name, arguments)

    @property
    def index(self) -> int:
        return self._index

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def function(self) -> FakeStreamFunction | None:
        return self._function


class FakeStreamFunction:
    def __init__(self, name: str | None, arguments: str | None) -> None:
        self._name: str | None = name
        self._arguments: str | None = arguments

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def arguments(self) -> str | None:
        return self._arguments


if __name__ == "__main__":
    pytest.main()
