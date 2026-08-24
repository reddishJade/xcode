"""流式事件解码单元测试。"""

from __future__ import annotations

from xcode.ai.events import ReasoningDelta, TextDelta, ToolCallEvent, UsageUpdate
from xcode.ai.providers._stream import (
    chat_stream_to_events,
    parse_tool_arguments,
)


class TestParseToolArguments:
    def test_valid_json(self) -> None:
        result = parse_tool_arguments('{"city": "Beijing"}')
        assert result == {"city": "Beijing"}

    def test_empty_object(self) -> None:
        result = parse_tool_arguments("{}")
        assert result == {}

    def test_empty_string_falls_back_to_empty_object(self) -> None:
        result = parse_tool_arguments("")
        assert result == {}

    def test_invalid_json(self) -> None:
        result = parse_tool_arguments("{broken}")
        assert "__invalid_tool_arguments__" in result


def _chunk(
    text: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict] | None = None,
    usage: object = None,
) -> object:
    class Delta:
        pass

    class Choice:
        pass

    class Chunk:
        pass

    delta = Delta()
    delta.content = text
    delta.reasoning_content = reasoning
    delta.tool_calls = tool_calls

    choice = Choice()
    choice.delta = delta

    chunk = Chunk()
    chunk.usage = usage
    chunk.choices = [choice]
    return chunk


def _usage(prompt: int = 0, completion: int = 0) -> object:
    class Usage:
        prompt_tokens = prompt
        completion_tokens = completion

    return Usage()


class TestChatStreamToEvents:
    def test_text_stream(self) -> None:
        chunks = [_chunk(text="Hello"), _chunk(text=" World")]
        events = list(chat_stream_to_events(chunks))
        texts = "".join(e.chunk for e in events if isinstance(e, TextDelta))
        assert texts == "Hello World"

    def test_reasoning_stream(self) -> None:
        chunks = [_chunk(reasoning="think"), _chunk(reasoning="ing")]
        events = list(chat_stream_to_events(chunks))
        reasons = "".join(e.chunk for e in events if isinstance(e, ReasoningDelta))
        assert reasons == "thinking"

    def test_tool_call_aggregation(self) -> None:
        class Func:
            def __init__(self, name: str | None = None, arguments: str | None = None):
                self.name = name
                self.arguments = arguments

        class Call:
            def __init__(
                self,
                index: int,
                id: str | None = None,
                name: str | None = None,
                arguments: str | None = None,
            ):
                self.index = index
                self.id = id
                self.function = Func(name, arguments) if name or arguments else None

        chunks = [
            _chunk(
                tool_calls=[Call(index=0, id="call1")],
            ),
            _chunk(
                tool_calls=[Call(index=0, name="get_weather")],
            ),
            _chunk(
                tool_calls=[Call(index=0, arguments='{"city":')],
            ),
            _chunk(
                tool_calls=[Call(index=0, arguments='"Beijing"}')],
            ),
        ]
        events = list(chat_stream_to_events(chunks))
        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].calls[0].name == "get_weather"
        assert tool_events[0].calls[0].input == {"city": "Beijing"}

    def test_usage_update(self) -> None:
        chunks = [_chunk(text="hi", usage=_usage(prompt=10, completion=20))]
        events = list(chat_stream_to_events(chunks))
        usage_events = [e for e in events if isinstance(e, UsageUpdate)]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 10
        assert usage_events[0].output_tokens == 20

    def test_empty_stream(self) -> None:
        events = list(chat_stream_to_events([]))
        assert events == []

    def test_mixed_events(self) -> None:
        chunks = [
            _chunk(reasoning="think"),
            _chunk(text="Hello"),
            _chunk(usage=_usage(prompt=5, completion=10)),
            _chunk(text=" World"),
        ]
        events = list(chat_stream_to_events(chunks))
        assert any(isinstance(e, ReasoningDelta) for e in events)
        assert any(isinstance(e, TextDelta) for e in events)
        assert any(isinstance(e, UsageUpdate) for e in events)
