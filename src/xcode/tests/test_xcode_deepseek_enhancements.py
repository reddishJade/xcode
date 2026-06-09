from __future__ import annotations

import asyncio
import unittest
from typing import Any, cast
from unittest.mock import patch

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
from xcode.ai.types import StreamOptions
from xcode.harness.skills import ToolSpec


class XcodeDeepSeekEnhancementsTests(unittest.TestCase):
    def test_reasoning_content_and_usage_are_extracted_in_streaming(self) -> None:
        client = FakeOpenAIClient(
            stream_chunks=[
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

        self.assertIsInstance(events[0], ReasoningDelta)
        self.assertEqual(cast(ReasoningDelta, events[0]).chunk, "Thinking 1")
        self.assertIsInstance(events[1], TextDelta)
        self.assertEqual(cast(TextDelta, events[1]).chunk, "Hello")
        self.assertIsInstance(events[2], UsageUpdate)
        self.assertEqual(cast(UsageUpdate, events[2]).input_tokens, 120)
        self.assertEqual(cast(UsageUpdate, events[2]).output_tokens, 0)
        self.assertEqual(provider.metrics["prompt_cache_hit_tokens"], 100)

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

        self.assertIsNone(converted[0]["content"])
        self.assertEqual(converted[1]["role"], "tool")
        self.assertEqual(converted[1]["tool_call_id"], "t1")
        self.assertEqual(converted[1]["content"], "result_text")

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
        self.assertTrue(func["strict"])
        params = func["parameters"]
        self.assertEqual(params["additionalProperties"], False)
        self.assertEqual(sorted(params["required"]), ["items", "text"])
        self.assertNotIn("minLength", params["properties"]["text"])
        self.assertNotIn("maxLength", params["properties"]["text"])
        self.assertNotIn("minItems", params["properties"]["items"])
        self.assertNotIn("maxItems", params["properties"]["items"])
        self.assertEqual(params["properties"]["text"]["type"], "string")
        self.assertEqual(params["properties"]["items"]["type"], ["array", "null"])

    def test_multi_chunk_tool_calls_streaming_concatenation(self) -> None:
        client = FakeOpenAIClient(
            stream_chunks=[
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

        self.assertIsInstance(events[-1], ToolCallEvent)
        ready_call = cast(ToolCallEvent, events[-1])
        self.assertEqual(ready_call.calls[0].name, "echo")
        self.assertEqual(ready_call.calls[0].input, {"text": "hi"})

    def test_stream_options_injection_via_public_entry(self) -> None:
        """验证 StreamOptions 通过 provider.stream() 注入到请求。"""

        async def run_test():
            captured_params: dict[str, Any] = {}

            def capture_create(**kwargs):
                captured_params.update(kwargs)
                return iter([FakeStreamChunk(content="ok")])

            client = FakeOpenAIClient(stream_chunks=[])
            client.chat.completions.create = capture_create
            provider = DeepSeekProvider(
                api_key="ds-key",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                client=client,
            )

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

            self.assertEqual(captured_params.get("api_key"), "override-key")
            extra_headers = captured_params.get("extra_headers", {})
            self.assertEqual(extra_headers.get("X-Custom"), "test-header")
            self.assertEqual(extra_headers.get("x-session-id"), "test-session-123")
            self.assertTrue(len(events) > 0)

        asyncio.run(run_test())

    def test_response_format_passed_to_request(self) -> None:
        """验证 response_format 传递到实际请求参数。"""
        captured_params: dict[str, Any] = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return iter([FakeStreamChunk(content="ok")])

        client = FakeOpenAIClient(stream_chunks=[])
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            client=client,
        )

        with patch.object(client.chat.completions, "create", capture_create):
            events = list(
                provider._stream_sync(
                    [{"role": "user", "content": "Hi"}],
                    (),
                    response_format={"type": "json_object"},
                )
            )

        self.assertEqual(
            captured_params.get("response_format"), {"type": "json_object"}
        )
        # 验证 JSON mode 防御逻辑注入了 'json' 关键字
        messages = captured_params.get("messages", [])
        self.assertTrue(len(messages) > 0)
        content = messages[0].get("content", "")
        self.assertIn("json", content.lower())
        self.assertTrue(len(events) > 0)


class FakeOpenAIClient:
    def __init__(self, stream_chunks=None) -> None:
        self.chat = FakeChat(stream_chunks)


class FakeChat:
    def __init__(self, stream_chunks) -> None:
        self.completions = FakeCompletions(stream_chunks)


class FakeCompletions:
    def __init__(self, stream_chunks) -> None:
        self.stream_chunks = stream_chunks

    def create(self, **kwargs):
        return iter(self.stream_chunks or [])


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
    unittest.main()
