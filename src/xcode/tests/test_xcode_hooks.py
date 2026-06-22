from __future__ import annotations

from typing import Any

from xcode.harness.observability import HookManager, HookRecord
from xcode.harness.observability import PreToolEvent, PostToolEvent
from xcode.harness.observability.hooks import BeforeProviderRequestEvent
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig, GateConfig

from xcode.tests.fixtures import FakeProvider
from xcode.ai.events import (
    ProviderEvent,
    TextDelta,
    FinalMessage,
    ToolCallEvent,
    ToolCall,
)
import pytest

EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


class XcodeHookTests:
    def test_typed_subscribers_receive_harness_events(self) -> None:
        seen: list[tuple[str, str]] = []
        hooks = HookManager()

        def record_pre(event) -> None:
            assert isinstance(event, PreToolEvent)
            seen.append((event.type, event.tool))

        def record_post(event) -> None:
            assert isinstance(event, PostToolEvent)
            seen.append((event.type, event.output))

        hooks.subscribe("pre_tool", record_pre)
        hooks.subscribe("post_tool", record_post)

        hooks.emit(HookRecord("pre_tool", tool="echo", input="hi"))
        hooks.emit(HookRecord("post_tool", tool="echo", output="done"))

        assert seen == [("pre_tool", "echo"), ("post_tool", "done")]

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
            registry=(
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            gate=GateConfig(hook_manager=hooks),
        )

        agent.run("go")

        assert seen == [("pre_tool", "echo"), ("post_tool", "hi")]

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
            registry=(ToolSpec("boom", "Boom.", "empty", fail, schema=EMPTY_SCHEMA),),
            gate=GateConfig(hook_manager=hooks),
        )

        result = agent.run("go")

        assert seen == ["Tool error: bad"]
        assert "Tool error: bad" in result.messages[2]["content"][0]["content"]

    def test_before_provider_request_includes_prompt_audit_metadata(self) -> None:
        seen: list[BeforeProviderRequestEvent] = []
        hooks = HookManager()

        def record(event: Any) -> None:
            assert isinstance(event, BeforeProviderRequestEvent)
            seen.append(event)

        hooks.subscribe("before_provider_request", record)
        provider = FakeProvider(
            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")]
        )
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            gate=GateConfig(hook_manager=hooks),
            runtime=AgentRuntimeConfig(
                runtime_context_provider=lambda _question: [
                    "<runtime>context</runtime>"
                ],
            ),
        )

        agent.run("go")

        assert len(seen) == 1
        event = seen[0]
        assert event.messages[0]["role"] == "system"
        assert "prompt_version" in event.metadata
        assert str(event.metadata["prompt_version"]).startswith("prompt:")
        assert "prompt_sha256" in event.metadata
        system_prompt_bytes = event.metadata["system_prompt_bytes"]
        assert isinstance(system_prompt_bytes, int)
        assert system_prompt_bytes > 0
        assert event.tools[0]["name"] == "echo"


if __name__ == "__main__":
    pytest.main()
