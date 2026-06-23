from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

from xcode.ai.providers.codec import (
    to_chat_messages,
    to_chat_tool,
)
from xcode.ai.providers.deepseek import DeepSeekProvider
from xcode.ai.events import (
    TextDelta,
    ReasoningDelta,
    UsageUpdate,
    ToolCallEvent,
)
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.skills import ToolSpec
import pytest


def _make_mock_client(chunks: list | None = None) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks or [])
    return client


class XcodeDeepSeekEnhancementsTests:
    def test_reasoning_content_and_usage_are_extracted_in_streaming(self) -> None:
        client = _make_mock_client(
            [
                FakeStreamChunk(reasoning_content="Thinking 1"),
                FakeStreamChunk(content="Hello"),
                FakeStreamChunk(usage=FakeUsage(100, 20)),
            ]
        )
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-reasoner",
            thinking=True,
            client=client,
        )

        events = list(provider._stream_sync([{"role": "user", "content": "Hi"}], ()))

        assert isinstance(events[0], ReasoningDelta)
        assert cast(ReasoningDelta, events[0]).chunk == "Thinking 1"
        assert isinstance(events[1], TextDelta)
        assert cast(TextDelta, events[1]).chunk == "Hello"
        assert isinstance(events[2], UsageUpdate)
        assert cast(UsageUpdate, events[2]).input_tokens == 120
        assert cast(UsageUpdate, events[2]).output_tokens == 0
        assert provider.metrics["prompt_cache_hit_tokens"] == 100

    def test_reasoning_content_history_cleanup_new_turn(self) -> None:
        messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "think",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "test", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": "result_text",
            },
        ]

        converted = to_chat_messages(messages)

        assert converted[0]["content"] is None
        assert converted[1]["role"] == "tool"
        assert converted[1]["tool_call_id"] == "t1"
        assert converted[1]["content"] == "result_text"

    def test_strict_tool_schema_conversions(self) -> None:
        tool = ToolSpec(
            name="strict_test",
            description="Strict testing.",
            input_hint="empty",
            handler=lambda _data: "",
            schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 10},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
            },
        )

        encoded = to_chat_tool(tool.name, tool.description, tool.schema, strict=True)

        func = encoded["function"]
        assert func["strict"]
        params = func["parameters"]
        assert not params["additionalProperties"]
        assert sorted(params["required"]) == ["items", "text"]
        assert "minLength" not in params["properties"]["text"]
        assert "maxLength" not in params["properties"]["text"]
        assert "minItems" not in params["properties"]["items"]
        assert "maxItems" not in params["properties"]["items"]
        assert params["properties"]["text"]["type"] == "string"
        assert params["properties"]["items"]["type"] == ["array", "null"]

    def test_multi_chunk_tool_calls_streaming_concatenation(self) -> None:
        client = _make_mock_client(
            [
                FakeStreamChunk(
                    tool_call=FakeStreamToolCall(
                        index=0,
                        call_id="call-1",
                        name="echo",
                        arguments='{"text": ',
                    )
                ),
                FakeStreamChunk(
                    tool_call=FakeStreamToolCall(
                        index=0,
                        arguments='"hi"}',
                    )
                ),
            ]
        )
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            thinking=False,
            client=client,
        )

        events = list(
            provider._stream_sync([{"role": "user", "content": "Run tool"}], ())
        )

        assert isinstance(events[-1], ToolCallEvent)
        ready_call = cast(ToolCallEvent, events[-1])
        assert ready_call.calls[0].name == "echo"
        assert ready_call.calls[0].input == {"text": "hi"}

    def test_stream_sends_tool_definition_parameters(self) -> None:
        """验证 DeepSeek 请求使用 ToolDefinition.parameters 作为工具参数。"""
        client = _make_mock_client([FakeStreamChunk(content="ok")])
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            thinking=False,
            client=client,
        )
        tool = ToolDefinition(
            name="read_file",
            description="Read a file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

        list(provider._stream_sync([{"role": "user", "content": "Read"}], (tool,)))

        sent_tool = client.chat.completions.create.call_args.kwargs["tools"][0]
        assert sent_tool["function"]["parameters"] == tool.parameters
        assert "input" not in sent_tool["function"]["parameters"]["properties"]

    def test_stream_options_injection_via_public_entry(self) -> None:
        """验证 StreamOptions 通过 provider.stream() 注入到请求。"""
        client = _make_mock_client([FakeStreamChunk(content="ok")])

        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            client=client,
        )

        async def run_test():
            options = StreamOptions(
                headers={"X-Custom": "test-header"},
                session_id="test-session-123",
                api_key="override-key",
            )
            events = [
                ev
                async for ev in provider.stream(
                    [{"role": "user", "content": "Hi"}], [], options=options
                )
            ]
            return events

        events = asyncio.run(run_test())
        kwargs = client.chat.completions.create.call_args.kwargs

        assert kwargs.get("model") == "deepseek-chat"
        assert kwargs.get("api_key") == "override-key"
        extra_headers = kwargs.get("extra_headers", {})
        assert extra_headers.get("X-Custom") == "test-header"
        assert extra_headers.get("x-session-id") == "test-session-123"
        assert len(events) > 0

    def test_response_format_passed_to_request(self) -> None:
        """验证 response_format 传递到实际请求参数。"""
        client = _make_mock_client([FakeStreamChunk(content="ok")])

        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            response_format={"type": "json_object"},
            client=client,
        )

        events = list(
            provider._stream_sync(
                [{"role": "user", "content": "Hi"}],
                (),
            )
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs.get("response_format") == {"type": "json_object"}
        messages = kwargs.get("messages", [])
        assert len(messages) > 0
        content = messages[0].get("content", "")
        assert "json" in content.lower()
        assert len(events) > 0


class FakeStreamChunk:
    def __init__(
        self, content=None, reasoning_content=None, usage=None, tool_call=None
    ) -> None:
        self.choices = (
            [FakeStreamChoice(content, reasoning_content, tool_call)]
            if usage is None
            else []
        )
        self.usage = usage


class FakeStreamChoice:
    def __init__(self, content, reasoning_content, tool_call=None) -> None:
        self.delta = FakeStreamDelta(content, reasoning_content, tool_call)


class FakeStreamDelta:
    def __init__(self, content, reasoning_content, tool_call=None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = [tool_call] if tool_call is not None else []


class FakeFunction:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


class FakeStreamToolCall:
    def __init__(self, index, call_id=None, name=None, arguments=None) -> None:
        self.index = index
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeUsage:
    def __init__(self, prompt_cache_hit_tokens, prompt_cache_miss_tokens) -> None:
        self.prompt_cache_hit_tokens = prompt_cache_hit_tokens
        self.prompt_cache_miss_tokens = prompt_cache_miss_tokens
        self.prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens
        self.completion_tokens = 0


if __name__ == "__main__":
    pytest.main()
