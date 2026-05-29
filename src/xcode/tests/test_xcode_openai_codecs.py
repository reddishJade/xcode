from __future__ import annotations

import unittest
from typing import cast

from xcode.agent.messages import convert_to_llm
from xcode.agent.types import AssistantMessage, ToolCallBlock
from xcode.harness.agent_runtime.events import ReasoningDelta, TextDelta, ToolCallReady
from xcode.ai.providers.codec import (
    chat_stream_to_events,
    to_chat_tool,
    to_openai_messages,
)
from xcode.harness.skills import ToolSpec


class OpenAIToolCodecTest(unittest.TestCase):
    def test_tool_schema_uses_explicit_schema(self) -> None:
        tool = ToolSpec(
            "echo",
            "Echo input.",
            'JSON: {"text":"..."}',
            lambda value: value,
            schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        encoded = to_chat_tool(tool.name, tool.description, tool.schema)

        self.assertEqual(encoded["function"]["parameters"], tool.schema)

    def test_tool_results_convert_to_openai_tool_messages(self) -> None:
        messages = to_openai_messages(
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

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[1]["role"], "tool")

    def test_tool_call_arguments_are_serialized_for_chat_api(self) -> None:
        messages = to_openai_messages(
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
        self.assertIsInstance(arguments, str)
        self.assertEqual(arguments, '{"path": "src/xcode"}')

    def test_reasoning_content_is_preserved_for_thinking_mode(self) -> None:
        messages = to_openai_messages(
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

        self.assertEqual(messages[0]["reasoning_content"], "private reasoning")

    def test_empty_reasoning_content_is_preserved_for_tool_calls(self) -> None:
        raw_messages = convert_to_llm(
            [
                AssistantMessage(
                    content=[
                        ToolCallBlock(
                            id="t1",
                            name="grep_search",
                            arguments={"input": "skill"},
                        )
                    ],
                    reasoning_content="",
                )
            ]
        )

        messages = to_openai_messages(raw_messages)

        self.assertEqual(messages[0]["reasoning_content"], "")


class OpenAIStreamCodecTest(unittest.TestCase):
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

        self.assertIsInstance(events[0], TextDelta)
        first_text = cast(TextDelta, events[0])
        self.assertEqual(first_text.chunk, "he")
        self.assertIsInstance(events[-1], ToolCallReady)
        final_call = cast(ToolCallReady, events[-1])
        self.assertEqual(final_call.calls[0].input, {"text": "hi"})

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
        self.assertEqual(len(events), 3)
        self.assertIsInstance(events[0], ReasoningDelta)
        first_reasoning = cast(ReasoningDelta, events[0])
        self.assertEqual(first_reasoning.chunk, "I am thinking")
        self.assertIsInstance(events[1], ReasoningDelta)
        second_reasoning = cast(ReasoningDelta, events[1])
        self.assertEqual(second_reasoning.chunk, " deeply")
        self.assertIsInstance(events[2], TextDelta)
        final_text = cast(TextDelta, events[2])
        self.assertEqual(final_text.chunk, "Hello")


class FakeStreamChunk:
    def __init__(self, tool_call=None, content=None, reasoning_content=None) -> None:
        self.choices = [FakeStreamChoice(content, tool_call, reasoning_content)]


class FakeStreamChoice:
    def __init__(self, content, tool_call, reasoning_content=None) -> None:
        self.delta = FakeStreamDelta(content, tool_call, reasoning_content)


class FakeStreamDelta:
    def __init__(self, content, tool_call, reasoning_content=None) -> None:
        self.content = content
        self.tool_calls = [tool_call] if tool_call is not None else []
        self.reasoning_content = reasoning_content


class FakeStreamToolCall:
    def __init__(self, index, call_id=None, name=None, arguments=None) -> None:
        self.index = index
        self.id = call_id
        self.function = FakeStreamFunction(name, arguments)


class FakeStreamFunction:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


if __name__ == "__main__":
    unittest.main()
