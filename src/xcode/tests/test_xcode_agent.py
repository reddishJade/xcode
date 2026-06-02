from __future__ import annotations

from typing import Any
import unittest

from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.messages import convert_to_llm
from xcode.agent.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from xcode.ai.events import Message, TextDelta, ToolCall, ToolCallEvent
from xcode.ai.types import ToolDefinition


class TextProvider:
    def __init__(self) -> None:
        self.messages: list[Message] | None = None
        self.tools: list[ToolDefinition] | None = None

    async def stream(self, messages: list[Message], tools: list[ToolDefinition]):
        self.messages = messages
        self.tools = tools
        yield TextDelta("ok")


class ToolProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[Message]] = []

    async def stream(self, messages: list[Message], tools: list[ToolDefinition]):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            yield ToolCallEvent([ToolCall("call-1", "echo", {"text": "hello"})])
            return
        yield TextDelta("done")


class EchoTool:
    name = "echo"
    label = "Echo"
    description = "Echo text."
    parameters = {"type": "object"}
    execution_mode = "sequential"

    def __init__(self) -> None:
        self.seen_signal: Any | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update=None,
    ) -> AgentToolResult[None]:
        self.seen_signal = signal
        return AgentToolResult(content=[TextContent(text=str(params["text"]))])


class AgentLoopContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_text_from_injected_provider(self) -> None:
        provider = TextProvider()
        events: list[AgentEvent] = []

        messages = await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
            ),
            emit=events.append,
        )

        self.assertEqual(provider.messages, [{"role": "user", "content": "hello"}])
        self.assertEqual(provider.tools, [])
        final = messages[-1]
        self.assertIsInstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        self.assertEqual(final.content, [TextContent(text="ok")])
        self.assertEqual(events[-1].type, "agent_end")

    async def test_executes_tools_without_harness_context(self) -> None:
        provider = ToolProvider()
        tool = EchoTool()
        events: list[AgentEvent] = []

        messages = await run_agent_loop(
            prompts=[UserMessage(content="use tool")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
            ),
            emit=events.append,
        )

        self.assertEqual(provider.calls, 2)
        self.assertIsNone(tool.seen_signal)
        self.assertTrue(any(isinstance(msg, ToolResultMessage) for msg in messages))
        self.assertEqual(provider.messages[-1][-1]["role"], "tool")
        final = messages[-1]
        self.assertIsInstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        self.assertEqual(final.content, [TextContent(text="done")])


if __name__ == "__main__":
    unittest.main()
