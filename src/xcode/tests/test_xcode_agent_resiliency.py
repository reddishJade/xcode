from __future__ import annotations

from typing import Any
from xcode.ai.events import (
    FinalMessage,
    ToolCall,
    ToolCallEvent,
    TextDelta,
)
from xcode.ai.types import StreamOptions
from xcode.harness.agent_runtime.structured import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.agent_runtime.fallback import _FallbackSwitchingProvider
from xcode.harness.config import AgentConfig
from xcode.harness.skills import ToolSpec
from xcode.tests.fixtures import FakeProvider
import pytest
PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

class XcodeAgentResiliencyTests:
    def test_diminishing_returns_continuation_circuit_breaker(self) -> None:
        # Mock provider returning max_tokens with small response blocks
        responses = iter(
            [
                [
                    TextDelta(chunk="a"),
                    FinalMessage(content="", stop_reason="max_tokens"),
                ],
                [
                    TextDelta(chunk="b"),
                    FinalMessage(content="", stop_reason="max_tokens"),
                ],
                [
                    TextDelta(chunk="c"),
                    FinalMessage(content="", stop_reason="max_tokens"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
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

        result = agent.run("test diminishing returns")

        assert "Diminishing Returns" in result.answer
        assert result.stopped_by_error

    def test_semantic_idle_failsafe_triggers_in_act_mode(self) -> None:
        # Define a read-only tool (not productive)
        read_tool = ToolSpec(
            name="read_file",
            description="read",
            input_hint="path",
            handler=lambda _data: "file content",
            schema=PATH_SCHEMA,
        )

        # Provider calls read_file consecutively with different paths to bypass repeated tool watchdog
        responses = iter(
            [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_1", name="read_file", input={"path": "a.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_2", name="read_file", input={"path": "b.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_3", name="read_file", input={"path": "c.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_4", name="read_file", input={"path": "d.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
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
        result = agent.run("test idle act", mode="act")
        assert result.stopped_by_watchdog
        assert result.watchdog_reason is not None
        assert result.watchdog_reason is not None
        assert "consecutive steps" in result.watchdog_reason.lower()

    def test_semantic_idle_failsafe_does_not_trigger_in_plan_mode(self) -> None:
        read_tool = ToolSpec(
            name="read_file",
            description="read",
            input_hint="path",
            handler=lambda _data: "file content",
            schema=PATH_SCHEMA,
        )

        responses = iter(
            [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_1", name="read_file", input={"path": "a.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_2", name="read_file", input={"path": "b.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_3", name="read_file", input={"path": "c.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="call_4", name="read_file", input={"path": "d.txt"}
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="tool_use"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
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
        assert result.answer == "done"

    def test_endpoint_session_level_fallback(self) -> None:
        primary_calls = 0
        fallback_calls = 0

        class OverloadedProvider:
            async def stream(
                self,
                messages,
                tools,
                options: StreamOptions | None = None,
                **kwargs: Any,
            ):
                nonlocal primary_calls
                primary_calls += 1
                if False:
                    yield None
                raise RuntimeError("529 Service Overloaded")

        class FallbackProvider:
            async def stream(
                self,
                messages,
                tools,
                options: StreamOptions | None = None,
                **kwargs: Any,
            ):
                nonlocal fallback_calls
                fallback_calls += 1
                yield TextDelta(chunk="fallback done")
                yield FinalMessage(content="", stop_reason="end_turn")

        agent = StructuredAgent(
            provider=OverloadedProvider(),
            registry=(),
            config=AgentConfig(max_steps=2),
            runtime=AgentRuntimeConfig(fallback_provider=FallbackProvider()),
        )

        result = agent.run("trigger fallback")
        assert result.answer == "fallback done"
        assert primary_calls == 3  # Overloaded failed 3 times
        assert fallback_calls == 1  # Fallback succeeded
        # After fallback, the wrapper should be using the fallback provider
        wrapper = agent.provider
        assert isinstance(wrapper, _FallbackSwitchingProvider)
        assert wrapper._using_fallback  # pyright: ignore[reportAttributeAccessIssue]

if __name__ == "__main__":
    pytest.main()
