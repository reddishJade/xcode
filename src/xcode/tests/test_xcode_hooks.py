from __future__ import annotations

import unittest

from xcode.harness.observability import HookManager
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent


from xcode.tests.fixtures import FakeProvider
from xcode.harness.agent_runtime.events import (
    TextDelta,
    FinalMessage,
    ToolCallReady,
    ToolCall,
)


class XcodeHookTests(unittest.TestCase):
    def test_hooks_fire_around_tool_execution(self) -> None:
        seen = []
        hooks = HookManager()
        hooks.register(
            "pre_tool", lambda record: seen.append((record.event, record.tool))
        )
        hooks.register(
            "post_tool", lambda record: seen.append((record.event, record.output))
        )
        from xcode.harness.agent_runtime.events import ProviderEvent

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallReady([ToolCall("x", "echo", "hi")]),
                FinalMessage("", "end_turn"),
            ],
            [TextDelta("done"), FinalMessage("", "end_turn")],
        ]
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(ToolSpec("echo", "Echo.", "text", lambda value: value),),
            hook_manager=hooks,
        )

        agent.run("go")

        self.assertEqual(seen, [("pre_tool", "echo"), ("post_tool", "hi")])

    def test_error_hook_fires_and_error_is_observed(self) -> None:
        seen = []
        hooks = HookManager()
        hooks.register("on_error", lambda record: seen.append(record.error))
        from xcode.harness.agent_runtime.events import ProviderEvent

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallReady([ToolCall("x", "boom", "")]),
                FinalMessage("", "end_turn"),
            ],
            [TextDelta("done"), FinalMessage("", "end_turn")],
        ]
        provider = FakeProvider(responses)

        def fail(_value: str) -> str:
            raise ValueError("bad")

        agent = StructuredAgent(
            provider=provider,
            registry=(ToolSpec("boom", "Boom.", "empty", fail),),
            hook_manager=hooks,
        )

        result = agent.run("go")

        self.assertEqual(seen, ["bad"])
        self.assertIn("tool error: bad", result.messages[2]["content"][0]["content"])


if __name__ == "__main__":
    unittest.main()
