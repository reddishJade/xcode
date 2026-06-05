from __future__ import annotations

import unittest

from xcode.harness.observability import HookManager, HookRecord
from xcode.harness.observability import PreToolEvent, PostToolEvent
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent


from xcode.tests.fixtures import FakeProvider
from xcode.ai.events import (
    ProviderEvent,
    TextDelta,
    FinalMessage,
    ToolCallEvent,
    ToolCall,
)


class XcodeHookTests(unittest.TestCase):
    def test_typed_subscribers_receive_harness_events(self) -> None:
        seen: list[tuple[str, str]] = []
        hooks = HookManager()

        def record_pre(event) -> None:
            self.assertIsInstance(event, PreToolEvent)
            seen.append((event.type, event.tool))

        def record_post(event) -> None:
            self.assertIsInstance(event, PostToolEvent)
            seen.append((event.type, event.output))

        hooks.subscribe("pre_tool", record_pre)
        hooks.subscribe("post_tool", record_post)

        hooks.emit(HookRecord("pre_tool", tool="echo", input="hi"))
        hooks.emit(HookRecord("post_tool", tool="echo", output="done"))

        self.assertEqual(seen, [("pre_tool", "echo"), ("post_tool", "done")])

    def test_hooks_fire_around_tool_execution(self) -> None:
        seen = []
        hooks = HookManager()
        hooks.register(
            "pre_tool", lambda record: seen.append((record.event, record.tool))
        )
        hooks.register(
            "post_tool", lambda record: seen.append((record.event, record.output))
        )
        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent([ToolCall("x", "echo", {"input": "hi"})]),
                FinalMessage("", "end_turn"),
            ],
            [TextDelta("done"), FinalMessage("", "end_turn")],
        ]
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(ToolSpec("echo", "Echo.", "text", lambda value: value["input"]),),
            hook_manager=hooks,
        )

        agent.run("go")

        self.assertEqual(seen, [("pre_tool", "echo"), ("post_tool", "hi")])

    def test_error_hook_fires_and_error_is_observed(self) -> None:
        seen = []
        hooks = HookManager()
        hooks.register("on_error", lambda record: seen.append(record.error))
        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent([ToolCall("x", "boom", {})]),
                FinalMessage("", "end_turn"),
            ],
            [TextDelta("done"), FinalMessage("", "end_turn")],
        ]
        provider = FakeProvider(responses)

        def fail(_value: dict) -> str:
            raise ValueError("bad")

        agent = StructuredAgent(
            provider=provider,
            registry=(ToolSpec("boom", "Boom.", "empty", fail),),
            hook_manager=hooks,
        )

        result = agent.run("go")

        self.assertEqual(seen, ["Tool error: bad"])
        self.assertIn("Tool error: bad", result.messages[2]["content"][0]["content"])


if __name__ == "__main__":
    unittest.main()
