from __future__ import annotations

import unittest

from xcode.ai.events import (
    FinalMessage,
    ToolCall,
    ToolCallEvent,
    TextDelta,
)
from xcode.harness.agent_runtime.structured import StructuredAgent
from xcode.harness.config import AgentConfig
from xcode.harness.skills import ToolSpec
from xcode.tests.fixtures import FakeProvider


class XcodeAgentResiliencyTests(unittest.TestCase):
    def test_diminishing_returns_continuation_circuit_breaker(self) -> None:
        # Mock provider returning max_tokens with small response blocks
        responses = iter(
            [
                [TextDelta("a"), FinalMessage("", "max_tokens")],
                [TextDelta("b"), FinalMessage("", "max_tokens")],
                [TextDelta("c"), FinalMessage("", "max_tokens")],
                [TextDelta("done"), FinalMessage("", "end_turn")],
            ]
        )

        def factory(messages, _tools):
            return next(responses)

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(),
            config=AgentConfig(max_steps=5),
        )

        with self.assertRaises(RuntimeError) as ctx:
            agent.run("test diminishing returns")
        self.assertIn("Diminishing Returns", str(ctx.exception))

    def test_semantic_idle_failsafe_triggers_in_act_mode(self) -> None:
        # Define a read-only tool (not productive)
        read_tool = ToolSpec(
            name="read_file",
            description="read",
            input_hint="path",
            handler=lambda _data: "file content",
            risk="low",
        )

        # Provider calls read_file consecutively with different paths to bypass repeated tool watchdog
        responses = iter(
            [
                [
                    ToolCallEvent([ToolCall("call_1", "read_file", {"path": "a.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [
                    ToolCallEvent([ToolCall("call_2", "read_file", {"path": "b.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [
                    ToolCallEvent([ToolCall("call_3", "read_file", {"path": "c.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [
                    ToolCallEvent([ToolCall("call_4", "read_file", {"path": "d.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
            ]
        )

        def factory(messages, _tools):
            return next(responses)

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(read_tool,),
            config=AgentConfig(max_steps=6),
        )

        # Act mode should trigger Watchdog after 4 consecutive idle steps
        with self.assertRaises(RuntimeError) as ctx:
            agent.run("test idle act", mode="act")
        self.assertIn("Watchdog triggered: 4 consecutive steps", str(ctx.exception))

    def test_semantic_idle_failsafe_does_not_trigger_in_plan_mode(self) -> None:
        read_tool = ToolSpec(
            name="read_file",
            description="read",
            input_hint="path",
            handler=lambda _data: "file content",
            risk="low",
        )

        responses = iter(
            [
                [
                    ToolCallEvent([ToolCall("call_1", "read_file", {"path": "a.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [
                    ToolCallEvent([ToolCall("call_2", "read_file", {"path": "b.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [
                    ToolCallEvent([ToolCall("call_3", "read_file", {"path": "c.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [
                    ToolCallEvent([ToolCall("call_4", "read_file", {"path": "d.txt"})]),
                    FinalMessage("", "tool_use"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
            ]
        )

        def factory(messages, _tools):
            return next(responses)

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(read_tool,),
            config=AgentConfig(max_steps=6),
        )

        # Plan mode should NOT trigger Watchdog, allowing exploration
        result = agent.run("test idle plan", mode="plan")
        self.assertEqual(result.answer, "done")

    def test_endpoint_session_level_fallback(self) -> None:
        primary_calls = 0
        fallback_calls = 0

        class OverloadedProvider:
            async def stream(self, messages, tools):
                nonlocal primary_calls
                primary_calls += 1
                if False:
                    yield None
                raise RuntimeError("529 Service Overloaded")

        class FallbackProvider:
            async def stream(self, messages, tools):
                nonlocal fallback_calls
                fallback_calls += 1
                yield TextDelta("fallback done")
                yield FinalMessage("", "end_turn")

        agent = StructuredAgent(
            provider=OverloadedProvider(),
            registry=(),
            fallback_provider=FallbackProvider(),
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("trigger fallback")
        self.assertEqual(result.answer, "fallback done")
        self.assertEqual(primary_calls, 3)  # Overloaded failed 3 times
        self.assertEqual(fallback_calls, 1)  # Fallback succeeded
        self.assertEqual(
            agent.provider, agent.fallback_provider
        )  # Permanent session fallback


if __name__ == "__main__":
    unittest.main()
