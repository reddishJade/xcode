from __future__ import annotations

import unittest
from typing import Any

from xcode.harness.observability import HookManager, HookRecord
from xcode.harness.observability import PreToolEvent, PostToolEvent
from xcode.harness.observability.hooks import BeforeProviderRequestEvent
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
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="echo", input={"input": "hi"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")],
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
                ToolCallEvent(calls=[ToolCall(id="x", name="boom", input={})]),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")],
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

    def test_before_provider_request_includes_prompt_audit_metadata(self) -> None:
        seen: list[BeforeProviderRequestEvent] = []
        hooks = HookManager()

        def record(event: Any) -> None:
            self.assertIsInstance(event, BeforeProviderRequestEvent)
            seen.append(event)

        hooks.subscribe("before_provider_request", record)
        provider = FakeProvider(
            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")]
        )
        agent = StructuredAgent(
            provider=provider,
            registry=(ToolSpec("echo", "Echo.", "text", lambda value: value["input"]),),
            hook_manager=hooks,
            runtime_context_provider=lambda _question: ["<runtime>context</runtime>"],
        )

        agent.run("go")

        self.assertEqual(len(seen), 1)
        event = seen[0]
        self.assertEqual(event.messages[0]["role"], "system")
        self.assertIn("prompt_version", event.metadata)
        self.assertTrue(str(event.metadata["prompt_version"]).startswith("prompt:"))
        self.assertIn("prompt_sha256", event.metadata)
        self.assertGreater(event.metadata["system_prompt_bytes"], 0)
        self.assertEqual(event.tools[0]["name"], "echo")


if __name__ == "__main__":
    unittest.main()
