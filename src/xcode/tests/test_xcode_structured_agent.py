from __future__ import annotations

import asyncio
import json
import sys
import threading
import unittest
from unittest.mock import patch

from collections.abc import Iterator
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
from xcode.harness.agent_runtime.event_translation import FinalStructuredEvent
from xcode.agent.messages import UserMessage
from xcode.harness.skills import ToolSpec
from xcode.tests.fixtures import FakeProvider


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


class XcodeStructuredAgentTests(unittest.TestCase):
    def _assert_final_answer(self, event: StructuredAgentEvent, expected: str) -> None:
        """断言最终事件包含指定答案。"""
        self.assertIsInstance(event, FinalStructuredEvent)
        assert isinstance(event, FinalStructuredEvent)
        self.assertEqual(event.data.answer, expected)

    def test_chat_turn_still_uses_normal_runtime_boundary(self) -> None:
        seen_tools: list[list[str]] = []

        def factory(messages, tools):
            seen_tools.append([tool.name for tool in tools])
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("<git-preflight>", messages[0]["content"])
            return [
                TextDelta(chunk="你好，我是 Xcode。"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        tool = ToolSpec("read_file", "Read file.", "path", lambda _input: "content")
        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(tool,),
            runtime_context_provider=lambda _question: [
                "<git-preflight>dirty</git-preflight>"
            ],
        )

        result = agent.run("hello, who are you?")

        self.assertEqual(result.answer, "你好，我是 Xcode。")
        self.assertEqual(seen_tools, [["read_file"]])
        self.assertEqual(result.tool_calls, [])

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

        self.assertEqual(result.answer, "done")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.last_agent, "main")
        self.assertEqual(result.tool_calls, [])
        assert result.metrics is not None
        self.assertEqual(result.metrics["llm_calls"], 1)

    def test_follow_up_turn_receives_previous_messages(self) -> None:
        """短追问应看到上一轮 user 和 assistant 消息。"""
        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    TextDelta(chunk="AGENTS.md is 10000 bytes."),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="是，正好 10000 字节。"),
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

        list(agent.run_stream("AGENTS.md的字节数是多少？"))
        second_events = list(agent.run_stream("正好是10000？"))

        self.assertEqual(
            [message["role"] for message in seen_messages[1]],
            ["user", "assistant", "user"],
        )
        self.assertIn("AGENTS.md的字节数", str(seen_messages[1][0]["content"]))
        self.assertIn("10000 bytes", str(seen_messages[1][1]["content"]))
        self.assertIn("正好是10000", str(seen_messages[1][2]["content"]))
        self.assertEqual(
            [event.type for event in second_events],
            ["message_start", "text_delta", "assistant", "turn_end", "final"],
        )

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

        self.assertEqual(result.answer, "restored")
        self.assertIsNotNone(result.run_state)
        assert result.run_state is not None
        self.assertEqual(result.run_state.current_mode, "act")
        self.assertEqual(
            [message["role"] for message in seen_messages[0]],
            ["user", "assistant", "user"],
        )

    def test_history_replacement_resets_provider_conversation_state(self) -> None:
        provider = ResettableFakeProvider()
        agent = StructuredAgent(provider=provider, registry=())

        agent.clear_history()
        agent.load_history([UserMessage(content="old")])
        agent.load_run_state(RunState(messages=[], current_mode="act"))

        self.assertEqual(provider.reset_count, 3)

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

        self.assertEqual(result.answer, "done")

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
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=3),
        )

        result = agent.run("go")

        self.assertEqual(result.answer, "finished")
        self.assertEqual(len(result.tool_calls), 2)
        assert result.metrics is not None
        self.assertEqual(result.metrics["tool_calls"], 2)
        self.assertEqual(result.messages[2]["content"][0]["tool_use_id"], "a")

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

        self.assertIn(
            "unknown tool: missing", result.messages[2]["content"][0]["content"]
        )
        self.assertEqual(result.messages[2]["content"][0]["status"], "error")
        self.assertEqual(result.answer, "saw error")

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

        self.assertTrue(result.stopped_by_limit)
        self.assertEqual(result.answer, "step limit reached")

    def test_watchdog_stops_repeated_tool_call(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda data: data["input"])
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

        self.assertTrue(result.stopped_by_watchdog)
        self.assertEqual(result.steps, 3)
        self.assertIn("watchdog stopped", result.answer)

    def test_watchdog_signature_stable_for_dict_input(self) -> None:
        tool = ToolSpec("echo", "Echo.", "text", lambda data: data["input"])
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

        self.assertTrue(result.stopped_by_watchdog)
        self.assertIn("watchdog stopped", result.answer)

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
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=5),
        )

        result = agent.run("inspect")

        self.assertEqual(result.answer, "done")

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
            runtime_context_provider=lambda _question: [
                "<skill>Review workflow.</skill>"
            ],
        )

        result = agent.run("please review this")

        self.assertEqual(result.answer, "done")
        self.assertEqual(seen[0]["role"], "system")
        self.assertIn("Review workflow", seen[0]["content"])

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
                ),
                ToolSpec(
                    "edit_file",
                    "Edit.",
                    "json",
                    edit_handler,
                    risk="high",
                ),
            ),
            config=AgentConfig(execution_mode="plan", max_steps=1),
        )

        result = agent.run("plan")

        self.assertEqual(seen_tools, [["read_file"]])
        self.assertEqual(called, [])
        self.assertIn(
            "unknown tool: edit_file",
            result.messages[3]["content"][0]["content"],
        )

    def test_review_mode_allows_git_diff_but_denies_other_bash(self) -> None:
        outputs = []

        def bash(data):
            outputs.append(data)
            return "diff"

        responses: Iterator[list[ProviderEvent]] = iter(
            [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="a",
                                name="bash",
                                input={"command": "git diff --stat"},
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="b",
                                name="bash",
                                input={"command": "python script.py"},
                            )
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
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(
                ToolSpec(
                    "bash",
                    "Shell.",
                    "json",
                    bash,
                    risk_evaluator=lambda _value: "allow",
                ),
            ),
            config=AgentConfig(execution_mode="review", max_steps=3),
        )

        result = agent.run("review")

        self.assertEqual(outputs, [{"command": "git diff --stat"}])
        tool_results = [
            block["content"]
            for message in result.messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        self.assertTrue(
            any(
                "requires approval" in item or "permission denied" in item
                for item in tool_results
            )
        )

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
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=(tool,),
            config=AgentConfig(max_steps=3),
        )

        events = list(agent.run_stream("go"))

        self.assertEqual(
            [event.type for event in events],
            [
                "message_start",
                "assistant",
                "tool_use",
                "tool_result",
                "turn_end",
                "text_delta",
                "assistant",
                "turn_end",
                "final",
            ],
        )
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

        self.assertEqual(
            [event.type for event in events],
            [
                "message_start",
                "text_delta",
                "text_delta",
                "assistant",
                "turn_end",
                "final",
            ],
        )
        self.assertEqual(events[1].data, "he")
        self._assert_final_answer(events[-1], "hello")

    def test_run_stream_uses_windows_selector_worker(self) -> None:
        if not hasattr(asyncio, "SelectorEventLoop"):
            self.skipTest("SelectorEventLoop is unavailable")

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

        self.assertTrue(selector_loop.called)
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

        self.assertEqual(result.answer, "async done")

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

        self.assertEqual(result.answer, "async done")

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

        self.assertEqual(
            [event.type for event in events],
            [
                "message_start",
                "text_delta",
                "text_delta",
                "assistant",
                "turn_end",
                "final",
            ],
        )
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
            with self.assertRaises(RuntimeError) as ctx:
                agent.run("go")
            return str(ctx.exception)

        message = asyncio.run(main())

        self.assertIn("use await StructuredAgent.run_async()", message)

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
                "first", "First.", "empty", first, read_only=True, concurrency_safe=True
            ),
            ToolSpec(
                "second",
                "Second.",
                "empty",
                second,
                read_only=True,
                concurrency_safe=True,
            ),
        )
        agent = StructuredAgent(
            provider=FakeProvider(lambda _m, _t: next(responses)),
            registry=tools,
            config=AgentConfig(max_steps=2, tool_workers=2),
        )

        result = agent.run("go")

        self.assertEqual(result.messages[2]["content"][0]["content"], "one")
        self.assertEqual(result.messages[3]["content"][0]["content"], "two")

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
            registry=(ToolSpec("boom", "Boom.", "empty", boom),),
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("go")

        self.assertEqual(result.messages[2]["content"][0]["status"], "error")
        self.assertIn("broken", result.messages[2]["content"][0]["content"])
        self.assertEqual(result.answer, "recovered")

    def test_cancelled_token_marks_tool_result_interrupted(self) -> None:
        token = CancellationToken()

        def factory(_messages, _tools) -> list[ProviderEvent]:
            token.cancel()
            return [
                ToolCallEvent(calls=[ToolCall(id="x", name="echo", input={})]),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        tool = ToolSpec("echo", "Echo.", "empty", lambda _data: "should not run")
        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(tool,),
            cancellation_token=token,
            config=AgentConfig(max_steps=2),
        )

        result = agent.run("go")

        self.assertEqual(result.messages[2]["content"][0]["status"], "interrupted")
        self.assertIn("interrupted", result.messages[2]["content"][0]["content"])

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
        self.assertEqual(result.answer, "part1 part2")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][-1]["role"], "user")
        self.assertEqual(calls[1][-1]["content"], "continue")

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

        self.assertTrue(result.stopped_by_error)
        self.assertIn("Diminishing Returns", result.answer)

    def test_transient_error_retry(self) -> None:
        import unittest.mock as mock

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
            self.assertEqual(result.answer, "success")
            self.assertEqual(len(calls), 3)
            self.assertEqual(mock_sleep.call_count, 2)

    def test_provider_error_retry_exhaustion_returns_fallback_message(self) -> None:
        import unittest.mock as mock

        def factory(_messages, _tools) -> list[ProviderEvent]:
            raise RuntimeError("provider down")

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )

        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            result = agent.run("hello")

        self.assertTrue(result.stopped_by_error)
        self.assertIn("I encountered an error.", result.answer)


if __name__ == "__main__":
    unittest.main()
