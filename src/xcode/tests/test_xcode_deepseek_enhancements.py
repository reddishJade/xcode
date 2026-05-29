from __future__ import annotations

import unittest
from typing import Any, cast

from xcode.ai.providers.codec import (
    to_openai_messages,
    to_chat_tool,
)
from xcode.ai.providers.deepseek import DeepSeekProvider
from xcode.harness.agent_runtime.events import (
    TextDelta,
    ReasoningDelta,
    UsageUpdate,
    ToolCallReady,
)
from xcode.harness.skills import ToolSpec


class XcodeDeepSeekEnhancementsTests(unittest.TestCase):
    def test_reasoning_content_is_returned_in_non_streaming(self) -> None:
        client = FakeOpenAIClient(
            content="Hello!",
            reasoning_content="I am thinking carefully.",
        )
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-reasoner",
            thinking=True,
            client=client,
        )

        res = provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(res["content"], "Hello!")
        self.assertEqual(res["reasoning_content"], "I am thinking carefully.")
        self.assertNotIn("temperature", client.chat.completions.kwargs)

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

    def test_invalid_parameters_removed_under_thinking_mode(self) -> None:
        client = FakeOpenAIClient(content="Hello")
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-reasoner",
            thinking=True,
            client=client,
        )

        provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.2,
        )

        kwargs = client.chat.completions.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("presence_penalty", kwargs)
        self.assertNotIn("frequency_penalty", kwargs)

    def test_tool_call_replay_does_not_stringify_none_content(self) -> None:
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

        converted = to_openai_messages(messages)

        self.assertIsNone(converted[0]["content"])
        self.assertEqual(converted[1]["role"], "tool")
        self.assertEqual(converted[1]["tool_call_id"], "t1")
        self.assertEqual(converted[1]["content"], "result_text")

    def test_strict_tool_schema_conversions(self) -> None:
        tool = ToolSpec(
            name="strict_test",
            description="Strict testing.",
            input_hint="empty",
            handler=lambda x: x,
            schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 10},
                    "items": {"type": "array", "minItems": 1},
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

    def test_json_output_mode_ensure_prompt_and_retry(self) -> None:
        client = FakeOpenAIClient(
            content="",  # first response empty content
            retry_content='{"ok": true}',  # second response has content
        )
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            thinking=False,
            client=client,
        )

        res = provider.complete(
            messages=[{"role": "user", "content": "Get output please."}],
            response_format={"type": "json_object"},
        )

        sent_messages = client.chat.completions.kwargs["messages"]
        self.assertIn("JSON", sent_messages[0]["content"])
        self.assertEqual(res["content"], '{"ok": true}')

    def test_json_output_mode_no_retry_on_tool_calls(self) -> None:
        client = FakeOpenAIClient(
            content="", tool_calls=[FakeToolCall("call-1", "test", "{}")]
        )
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            thinking=False,
            client=client,
        )

        provider.complete(
            messages=[{"role": "user", "content": "Get JSON please."}],
            response_format={"type": "json_object"},
        )

        self.assertEqual(client.chat.completions.call_count, 1)

    def test_reasoning_content_history_cleanup_new_turn(self) -> None:
        messages = [
            {"role": "user", "content": "Query 1"},
            {"role": "assistant", "content": "Ans 1", "reasoning_content": "Thought 1"},
            {"role": "user", "content": "Query 2"},
        ]
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="model",
            client=FakeOpenAIClient(),
        )

        cleaned = provider._clean_reasoning_content(messages)

        self.assertNotIn("reasoning_content", cleaned[1])

    def test_reasoning_content_history_cleanup_tool_loop(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Query 1"},
            {"role": "assistant", "content": "Ans 1", "reasoning_content": "Thought 1"},
            {"role": "user", "content": "Query 2"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Thought 2",
                "tool_calls": [],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "res"},
        ]
        provider = DeepSeekProvider(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="model",
            client=FakeOpenAIClient(),
        )

        cleaned = provider._clean_reasoning_content(messages)

        self.assertNotIn("reasoning_content", cleaned[1])
        self.assertEqual(cleaned[3]["reasoning_content"], "Thought 2")

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

        # Verify that the last event is ToolCallReady with correctly concatenated args {"text": "hi"}
        self.assertIsInstance(events[-1], ToolCallReady)
        ready_call = cast(ToolCallReady, events[-1])
        self.assertEqual(ready_call.calls[0].name, "echo")
        self.assertEqual(ready_call.calls[0].input, {"text": "hi"})


class FakeOpenAIClient:
    def __init__(
        self,
        content=None,
        reasoning_content=None,
        stream_chunks=None,
        retry_content=None,
        tool_calls=None,
    ) -> None:
        self.chat = FakeChat(
            content, reasoning_content, stream_chunks, retry_content, tool_calls
        )


class FakeChat:
    def __init__(
        self, content, reasoning_content, stream_chunks, retry_content, tool_calls
    ) -> None:
        self.completions = FakeCompletions(
            content, reasoning_content, stream_chunks, retry_content, tool_calls
        )


class FakeCompletions:
    def __init__(
        self, content, reasoning_content, stream_chunks, retry_content, tool_calls
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.stream_chunks = stream_chunks
        self.retry_content = retry_content
        self.tool_calls = tool_calls
        self.kwargs: dict[str, Any] = {}
        self.call_count = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.call_count += 1
        if kwargs.get("stream"):
            return iter(self.stream_chunks or [])

        content_to_use = self.content
        if self.call_count > 1 and self.retry_content is not None:
            content_to_use = self.retry_content

        return FakeResponse(content_to_use, self.reasoning_content, self.tool_calls)


class FakeResponse:
    def __init__(self, content, reasoning_content, tool_calls=None) -> None:
        self.choices = [FakeChoice(content, reasoning_content, tool_calls)]
        self.usage = None


class FakeChoice:
    def __init__(self, content, reasoning_content, tool_calls=None) -> None:
        self.message = FakeMessage(content, reasoning_content, tool_calls)


class FakeMessage:
    def __init__(self, content, reasoning_content, tool_calls=None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls if tool_calls is not None else []


class FakeToolCall:
    def __init__(self, call_id, name, arguments) -> None:
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeFunction:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


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
