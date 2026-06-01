from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from xcode.agent.agent import Agent
from xcode.agent.types import AssistantMessage, MessageUpdateEvent
from xcode.harness.agent_runtime.events import TextDelta


class StreamingProvider:
    def __init__(self, chunks: list[str], delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay

    async def stream(self, _tools, _context):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield TextDelta(chunk)


class XcodeAgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_and_remove_listener_are_explicit(self) -> None:
        model = SimpleNamespace(provider="fake", provider_obj=StreamingProvider(["ok"]))
        agent = Agent(model=model)
        seen: list[str] = []

        def record_event(event, _signal) -> None:
            seen.append(getattr(event, "type", ""))

        agent.add_listener(record_event)
        await agent.prompt("hello")
        self.assertIn("agent_end", seen)

        count_after_first_run = len(seen)
        agent.remove_listener(record_event)
        await agent.prompt("again")
        self.assertEqual(len(seen), count_after_first_run)

    async def test_abort_uses_xcode_cancellation_token(self) -> None:
        model = SimpleNamespace(
            provider="fake",
            provider_obj=StreamingProvider(["first", "second"], delay=0.01),
        )
        agent = Agent(model=model)

        def abort_on_first_delta(event, _signal) -> None:
            if isinstance(event, MessageUpdateEvent):
                agent.abort()

        agent.add_listener(abort_on_first_delta)
        prompt_task = asyncio.create_task(agent.prompt("stop"))
        await prompt_task
        await agent.wait_for_idle()

        self.assertFalse(agent.is_streaming)
        final_message = agent.messages[-1]
        self.assertIsInstance(final_message, AssistantMessage)
        assert isinstance(final_message, AssistantMessage)
        self.assertEqual(final_message.stop_reason, "aborted")
        self.assertEqual(agent.error_message, "interrupted by user")


if __name__ == "__main__":
    unittest.main()
