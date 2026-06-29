from __future__ import annotations

import asyncio
import json
import sys
import threading
import tempfile
import unittest.mock as mock

from unittest.mock import patch

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from xcode.harness.config import AgentConfig
from xcode.ai.events import (
    FinalMessage,
    TextDelta,
    ToolCallEvent,
    ToolCall,
    ProviderEvent,
)
from xcode.harness.agent_runtime import (
    CancellationToken,
    RunState,
    StructuredAgent,
    StructuredAgentEvent,
)
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.agent_runtime.events import FinalStructuredEvent
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.agent.messages import UserMessage
from xcode.agent.results import TerminationReason
from xcode.harness.memory import MemoryManager
from xcode.harness.skills import ToolSpec
from xcode.tests.fixtures import FakeProvider
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
TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}
PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}
ANY_OBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
}


class ResettableFakeProvider(FakeProvider):
    """记录会话状态重置次数的测试 provider。"""

    def __init__(self) -> None:
        events: list[ProviderEvent] = [
            TextDelta(chunk="ok"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        super().__init__(events)
        self.reset_count = 0

    def reset_conversation_state(self) -> None:
        """记录重置调用。"""
        self.reset_count += 1


class XcodeStructuredAgentTests:
    def _memory_runtime(
        self, project_root: Path
    ) -> tuple[AgentRuntimeConfig, MemoryManager]:
        manager = MemoryManager(
            project_root,
            user_memory_file=project_root / "home" / ".xcode" / "memory" / "MEMORY.md",
        )
        manager.memory_file.write_text(
            (
                "## Provider timeout retry\n"
                "- Context/Query: Provider timeout retry\n"
                "- Solution: Retry transient provider failures with backoff\n"
                "- Files: src/provider.py\n"
                "- Takeaways: Bound retries and preserve the root cause\n"
            ),
            encoding="utf-8",
        )
        runtime = AgentRuntimeConfig(
            project_root=project_root,
            runtime_context_provider=build_runtime_context_provider(
                project_root,
                (),
                memory_manager=manager,
            ),
            memory_manager=manager,
        )
        return runtime, manager

    def _assert_final_answer(self, event: StructuredAgentEvent, expected: str) -> None:
        """断言最终事件包含指定答案。"""
        assert isinstance(event, FinalStructuredEvent)
        assert isinstance(event, FinalStructuredEvent)
        assert event.data.answer == expected

    def test_chat_turn_still_uses_normal_runtime_boundary(self) -> None:
        seen_tools: list[list[str]] = []

        def factory(messages, tools):
            seen_tools.append([tool.name for tool in tools])
            assert messages[0]["role"] == "system"
            assert "<git-preflight>" in messages[0]["content"]
            return [
                TextDelta(chunk="Hello, I am Xcode."),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        tool = ToolSpec(
            "read_file",
            "Read file.",
            "path",
            lambda _input: "content",
            schema=PATH_SCHEMA,
        )
        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(tool,),
            runtime=AgentRuntimeConfig(
                runtime_context_provider=lambda _question: [
                    "<git-preflight>dirty</git-preflight>"
                ],
            ),
        )

        result = agent.run("hello, who are you?")

        assert result.answer == "Hello, I am Xcode."
        assert seen_tools == [["read_file"]]
        assert result.tool_calls == []

    def test_no_tool_call_returns_text(self) -> None:
        events: list[ProviderEvent] = [
            TextDelta(chunk="done"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(events),
            registry=(),
        )

        result = agent.run("hello")

        assert result.answer == "done"
        assert result.steps == 1
        assert result.last_agent == "main"
        assert result.tool_calls == []
        assert result.metrics is not None
        assert result.metrics["llm_calls"] == 1

    def test_follow_up_turn_receives_previous_messages(self) -> None:
        """短追问应看到上一轮 user 和 assistant 消息。"""
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    TextDelta(chunk="AGENTS.md is 10000 bytes."),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="Yes, exactly 10000 bytes."),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        seen_messages: list[list[dict[str, Any]]] = []

        def factory(
            messages: list[dict[str, Any]],
            _tools: list[Any],
        ) -> list[ProviderEvent]:
            seen_messages.append(messages)
            return next(responses)

        agent = StructuredAgent(provider=FakeProvider(factory), registry=())

        list(agent.run_stream("How many bytes is AGENTS.md?"))
        second_events = list(agent.run_stream("Exactly 10000?"))

        assert [message["role"] for message in seen_messages[1]] == [
            "user",
            "assistant",
            "user",
        ]
        assert "How many bytes is AGENTS.md" in str(seen_messages[1][0]["content"])
        assert "10000 bytes" in str(seen_messages[1][1]["content"])
        assert "Exactly 10000" in str(seen_messages[1][2]["content"])
        assert [event.type for event in second_events] == [
            "message_start",
            "text_delta",
            "assistant",
            "turn_end",
            "final",
        ]

    def test_load_run_state_restores_history_messages(self) -> None:
        seen_messages: list[list[dict[str, Any]]] = []

        def factory(
            messages: list[dict[str, Any]],
            _tools: list[Any],
        ) -> list[ProviderEvent]:
            seen_messages.append(messages)
            return [
                TextDelta(chunk="restored"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        run_state = RunState(
            messages=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ],
            current_mode="act",
        )
        agent = StructuredAgent(provider=FakeProvider(factory), registry=())

        agent.load_run_state(RunState.from_dict(run_state.to_dict()))
        result = agent.run("next question")

        assert result.answer == "restored"
        assert result.run_state is not None
        assert result.run_state is not None
        assert result.run_state.current_mode == "act"
        assert [message["role"] for message in seen_messages[0]] == [
            "user",
            "assistant",
            "user",
        ]

    def test_history_replacement_resets_provider_conversation_state(self) -> None:
        provider = ResettableFakeProvider()
        agent = StructuredAgent(provider=provider, registry=())

        agent.clear_history()
        agent.load_history([UserMessage(content="old")])
        agent.load_run_state(RunState(messages=[], current_mode="act"))

        assert provider.reset_count == 3

    def test_provider_events_drive_main_loop(self) -> None:
        events: list[ProviderEvent] = [
            TextDelta(chunk="done"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(events),
            registry=(),
        )

        result = agent.run("hello")

        assert result.answer == "done"

    def test_multiple_tool_calls_append_tool_results(self) -> None:
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(id="a", name="echo", input={"text": "one"}),
                            ToolCall(id="b", name="echo", input={"text": "two"}),
                        ]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="finished"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        tool = ToolSpec(
            name="echo",
            description="Echo input.",
            input_hint="JSON",
            handler=lambda data: json.dumps(data, ensure_ascii=False, sort_keys=True),
            read_only=True,
            concurrency_safe=True,
            schema=TEXT_SCHEMA,
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=3),
        )

        result = agent.run("go")

        assert result.answer == "finished"
        assert len(result.tool_calls) == 2
        assert result.metrics is not None
        assert result.metrics["tool_calls"] == 2
        assert result.messages[2]["content"][0]["tool_use_id"] == "a"

    def test_unknown_tool_is_returned_as_tool_result(self) -> None:
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(calls=[ToolCall(id="x", name="missing", input={})]),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="saw error"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(),
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("use missing")

        assert "unknown tool: missing" in result.messages[2]["content"][0]["content"]
        assert result.messages[2]["content"][0]["status"] == "error"
        assert result.answer == "saw error"

    def test_step_limit(self) -> None:
        events: list[ProviderEvent] = [
            ToolCallEvent(calls=[ToolCall(id="x", name="missing", input={})]),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(events),
            registry=(),
            config=AgentConfig(max_steps=1),
        )

        result = agent.run("loop")

        assert result.termination_reason is TerminationReason.STEP_LIMIT
        assert result.answer == "step limit reached"

    def test_watchdog_stops_repeated_tool_call(self) -> None:
        tool = ToolSpec(
            "echo",
            "Echo.",
            "text",
            lambda data: data["input"],
            schema=INPUT_SCHEMA,
        )
        agent = StructuredAgent(
            provider=FakeProvider(
                lambda _m, _t: cast(
                    list[ProviderEvent],
                    [
                        ToolCallEvent(
                            calls=[
                                ToolCall(id="x", name="echo", input={"input": "same"})
                            ]
                        ),
                        FinalMessage(content="", stop_reason="end_turn"),
                    ],
                )
            ),
            registry=(tool,),
            config=AgentConfig(max_steps=5, watchdog_repeated_tool_limit=2),
        )

        result = agent.run("loop")

        assert result.termination_reason is TerminationReason.WATCHDOG
        assert result.steps == 3
        assert "watchdog stopped" in result.answer

    def test_watchdog_signature_stable_for_dict_input(self) -> None:
        tool = ToolSpec(
            "echo",
            "Echo.",
            "text",
            lambda data: data["input"],
            schema=ANY_OBJECT_SCHEMA,
        )
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(
                        calls=[ToolCall(id="x", name="echo", input={"a": 1, "b": 2})]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    ToolCallEvent(
                        calls=[ToolCall(id="y", name="echo", input={"b": 2, "a": 1})]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=3, watchdog_repeated_tool_limit=1),
        )

        result = agent.run("test")

        assert result.termination_reason is TerminationReason.WATCHDOG
        assert "watchdog stopped" in result.answer

    def test_idle_watchdog_allows_successful_read_only_steps(self) -> None:
        mock_response_list: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[
                        ToolCall(
                            id=f"r{index}",
                            name="read_file",
                            input={"path": f"notes-{index}.md"},
                        )
                    ]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
            for index in range(4)
        ]
        last_event: list[ProviderEvent] = [
            TextDelta(chunk="done"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        mock_response_list.append(last_event)
        responses: Iterator[list[ProviderEvent]] = iter(mock_response_list)
        tool = ToolSpec(
            "read_file",
            "Read.",
            "path",
            lambda _data: "content",
            read_only=True,
            schema=PATH_SCHEMA,
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=5),
        )

        result = agent.run("inspect")

        assert result.answer == "done"

    def test_runtime_context_provider_injects_system_message(self) -> None:
        seen = []

        def factory(messages, _tools) -> list[ProviderEvent]:
            seen.append(messages[0])
            return [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
            runtime=AgentRuntimeConfig(
                runtime_context_provider=lambda _question: [
                    "<skill>Review workflow.</skill>"
                ],
            ),
        )

        result = agent.run("please review this")

        assert result.answer == "done"
        assert seen[0]["role"] == "system"
        assert "Review workflow" in seen[0]["content"]

    def test_plan_mode_exposes_read_tools_and_blocks_write_tools(self) -> None:
        seen_tools = []
        called = []

        def factory(_messages, tools) -> list[ProviderEvent]:
            seen_tools.append([tool.name for tool in tools])
            return [
                ToolCallEvent(
                    calls=[ToolCall(id="x", name="edit_file", input={"input": "hello"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        def edit_handler(data: dict) -> str:
            called.append(data["input"])
            return data["input"]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(
                ToolSpec(
                    "read_file",
                    "Read.",
                    "path",
                    lambda data: data["input"],
                    read_only=True,
                    schema=PATH_SCHEMA,
                ),
                ToolSpec(
                    "edit_file",
                    "Edit.",
                    "json",
                    edit_handler,
                    schema=INPUT_SCHEMA,
                ),
            ),
            config=AgentConfig(execution_mode="plan", max_steps=1),
        )

        result = agent.run("plan")

        assert seen_tools == [["read_file"]]
        assert called == []
        assert "unknown tool: edit_file" in result.messages[3]["content"][0]["content"]

    def test_run_stream_yields_tool_and_final_events(self) -> None:
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(
                        calls=[ToolCall(id="a", name="echo", input={"input": "hello"})]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        tool = ToolSpec(
            name="echo",
            description="Echo input.",
            input_hint="text",
            handler=lambda data: data["input"],
            schema=INPUT_SCHEMA,
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=3),
        )

        events = list(agent.run_stream("go"))

        assert [event.type for event in events] == [
            "message_start",
            "assistant",
            "tool_use",
            "tool_result",
            "turn_end",
            "text_delta",
            "assistant",
            "turn_end",
            "final",
        ]
        self._assert_final_answer(events[-1], "done")

    def test_run_stream_yields_text_delta_events(self) -> None:
        mock_events: list[ProviderEvent] = [
            TextDelta(chunk="he"),
            TextDelta(chunk="llo"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(mock_events),
            registry=(),
        )

        events = list(agent.run_stream("go"))

        assert [event.type for event in events] == [
            "message_start",
            "text_delta",
            "text_delta",
            "assistant",
            "turn_end",
            "final",
        ]
        assert events[1].data == "he"
        self._assert_final_answer(events[-1], "hello")

    def test_run_stream_uses_windows_selector_worker(self) -> None:
        if not hasattr(asyncio, "SelectorEventLoop"):
            pytest.skip("SelectorEventLoop is unavailable")

        mock_events: list[ProviderEvent] = [
            TextDelta(chunk="done"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(mock_events),
            registry=(),
        )

        with (
            patch.object(sys, "platform", "win32"),
            patch(
                "xcode.harness.agent_runtime.async_worker.asyncio.SelectorEventLoop",
                side_effect=asyncio.new_event_loop,
            ) as selector_loop,
        ):
            events = list(agent.run_stream("go"))

        assert selector_loop.called
        self._assert_final_answer(events[-1], "done")

    def test_run_stream_does_not_call_asyncio_run_in_bridge(self) -> None:
        mock_events: list[ProviderEvent] = [
            TextDelta(chunk="done"),
            FinalMessage(content="", stop_reason="end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(mock_events),
            registry=(),
        )

        with patch(
            "xcode.harness.agent_runtime.agent_helpers.asyncio.run",
            side_effect=AssertionError("asyncio.run should not be used by run_stream"),
        ):
            events = list(agent.run_stream("go"))

        self._assert_final_answer(events[-1], "done")

    def test_arun_returns_result_inside_event_loop(self) -> None:
        async def main():
            events: list[ProviderEvent] = [
                TextDelta(chunk="async done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
            agent = StructuredAgent(
                provider=FakeProvider(events),
                registry=(),
            )
            return await agent.arun("go")

        result = asyncio.run(main())

        assert result.answer == "async done"

    def test_run_async_returns_result_inside_event_loop(self) -> None:
        async def main():
            events: list[ProviderEvent] = [
                TextDelta(chunk="async done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
            agent = StructuredAgent(
                provider=FakeProvider(events),
                registry=(),
            )
            return await agent.run_async("go")

        result = asyncio.run(main())

        assert result.answer == "async done"

    def test_arun_stream_yields_events_inside_event_loop(self) -> None:
        async def main():
            events: list[ProviderEvent] = [
                TextDelta(chunk="he"),
                TextDelta(chunk="llo"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
            agent = StructuredAgent(
                provider=FakeProvider(events),
                registry=(),
            )
            stream_events: list[StructuredAgentEvent] = []
            async for event in agent.arun_stream("go"):
                stream_events.append(event)
            return stream_events

        events = asyncio.run(main())

        assert [event.type for event in events] == [
            "message_start",
            "text_delta",
            "text_delta",
            "assistant",
            "turn_end",
            "final",
        ]
        self._assert_final_answer(events[-1], "hello")

    def test_sync_api_rejects_active_event_loop(self) -> None:
        async def main():
            events: list[ProviderEvent] = [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
            agent = StructuredAgent(
                provider=FakeProvider(events),
                registry=(),
            )
            with pytest.raises(RuntimeError) as exc_info:
                agent.run("go")
            return str(exc_info.value)

        message = asyncio.run(main())

        assert "use await StructuredAgent.run_async()" in message

    def test_read_only_tools_run_in_threadpool(self) -> None:
        started_second = threading.Event()

        def first(_data: dict) -> str:
            if not started_second.wait(1):
                return "not parallel"
            return "one"

        def second(_data: dict) -> str:
            started_second.set()
            return "two"

        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(id="a", name="first", input={}),
                            ToolCall(id="b", name="second", input={}),
                        ]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        tools = (
            ToolSpec(
                "first",
                "First.",
                "empty",
                first,
                read_only=True,
                concurrency_safe=True,
                schema=EMPTY_SCHEMA,
            ),
            ToolSpec(
                "second",
                "Second.",
                "empty",
                second,
                read_only=True,
                concurrency_safe=True,
                schema=EMPTY_SCHEMA,
            ),
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=tools,
            config=AgentConfig(max_steps=2, tool_workers=2),
        )

        result = agent.run("go")

        assert result.messages[2]["content"][0]["content"] == "one"
        assert result.messages[3]["content"][0]["content"] == "two"

    def test_tool_exception_is_error_result_and_agent_continues(self) -> None:
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(calls=[ToolCall(id="x", name="boom", input={})]),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="recovered"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )

        def boom(_data: dict) -> str:
            raise RuntimeError("broken")

        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(ToolSpec("boom", "Boom.", "empty", boom, schema=EMPTY_SCHEMA),),
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("go")

        assert result.messages[2]["content"][0]["status"] == "error"
        assert "broken" in result.messages[2]["content"][0]["content"]
        assert result.answer == "recovered"

    def test_cancelled_token_marks_tool_result_interrupted(self) -> None:
        token = CancellationToken()

        def factory(_messages, _tools) -> list[ProviderEvent]:
            token.cancel()
            return [
                ToolCallEvent(calls=[ToolCall(id="x", name="echo", input={})]),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        tool = ToolSpec(
            "echo",
            "Echo.",
            "empty",
            lambda _data: "should not run",
            schema=EMPTY_SCHEMA,
        )
        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
            runtime=AgentRuntimeConfig(cancellation_token=token),
        )

        result = agent.run("go")

        assert result.messages[2]["content"][0]["status"] == "interrupted"
        assert "interrupted" in result.messages[2]["content"][0]["content"]

    def test_max_tokens_truncation_triggers_continuation(self) -> None:
        calls = []

        def factory(messages, tools) -> list[ProviderEvent]:
            calls.append(list(messages))
            if len(calls) == 1:
                return [
                    TextDelta(chunk="part1"),
                    FinalMessage(content="", stop_reason="max_tokens"),
                ]
            else:
                return [
                    TextDelta(chunk=" part2"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )
        result = agent.run("hello")
        assert result.answer == "part1 part2"
        assert len(calls) == 2
        assert calls[1][-1]["role"] == "user"
        assert calls[1][-1]["content"] == "continue"

    def test_low_token_continuation_circuit_breaker(self) -> None:
        def factory(messages, tools) -> list[ProviderEvent]:
            return [
                TextDelta(chunk="x"),
                FinalMessage(content="", stop_reason="max_tokens"),
            ]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )
        result = agent.run("hello")

        assert result.termination_reason is TerminationReason.PROVIDER_ERROR
        assert "Diminishing Returns" in result.answer

    def test_transient_error_retry(self) -> None:
        calls = []

        def factory(messages, tools) -> list[ProviderEvent]:
            calls.append(messages)
            if len(calls) < 3:
                raise RuntimeError("529 overloaded")
            return [
                TextDelta(chunk="success"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()) as mock_sleep:
            result = agent.run("hello")
            assert result.answer == "success"
            assert len(calls) == 3
            assert mock_sleep.call_count == 2

    def test_provider_error_retry_exhaustion_returns_fallback_message(self) -> None:
        def factory(_messages, _tools) -> list[ProviderEvent]:
            raise RuntimeError("provider down")

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )

        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            result = agent.run("hello")

        assert result.termination_reason is TerminationReason.PROVIDER_ERROR
        assert "I encountered an error." in result.answer

    def test_successful_run_persists_memory_adoption_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runtime, manager = self._memory_runtime(project_root)
            agent = StructuredAgent(
                provider=FakeProvider(
                    [
                        TextDelta(chunk="done"),
                        FinalMessage(content="", stop_reason="end_turn"),
                    ]
                ),
                registry=(),
                runtime=runtime,
            )

            result = agent.run("provider timeout retry")

            assert result.termination_reason is TerminationReason.COMPLETED
            record = manager.read_memory_records(layer="project")[0]
            assert record.injection_count == 1
            assert record.adoption_count == 1
            assert record.success_count == 1
            assert record.utility == 1.0
            trace_events = manager.drain_trace_events()
            assert any(event.type == "used" for event in trace_events)

    def test_provider_error_run_persists_memory_failure_without_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runtime, manager = self._memory_runtime(project_root)

            def factory(_messages, _tools) -> list[ProviderEvent]:
                raise RuntimeError("provider down")

            agent = StructuredAgent(
                provider=FakeProvider(factory),
                registry=(),
                runtime=runtime,
            )

            result = agent.run("provider timeout retry")

            assert result.termination_reason is TerminationReason.PROVIDER_ERROR
            record = manager.read_memory_records(layer="project")[0]
            assert record.injection_count == 1
            assert record.adoption_count == 0
            assert record.failure_count == 0
            assert record.last_outcome == "failure"
            trace_events = manager.drain_trace_events()
            assert all(event.type != "used" for event in trace_events)

    def test_ambiguous_empty_run_skips_memory_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runtime, manager = self._memory_runtime(project_root)
            agent = StructuredAgent(
                provider=FakeProvider([]),
                registry=(),
                runtime=runtime,
            )

            result = agent.run("provider timeout retry")

            assert result.termination_reason is TerminationReason.COMPLETED
            assert result.answer == ""
            record = manager.read_memory_records(layer="project")[0]
            assert record.injection_count == 0
            assert record.adoption_count == 0
            assert record.success_count == 0
            assert record.failure_count == 0
            assert record.last_outcome is None


if __name__ == "__main__":
    pytest.main()
