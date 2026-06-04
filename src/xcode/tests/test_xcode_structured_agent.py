from __future__ import annotations

import asyncio
import json
import sys
import threading
import unittest
from unittest.mock import patch

from typing import cast, Iterator
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
    StructuredAgent,
    StructuredAgentEvent,
)
from xcode.harness.skills import ToolSpec
from xcode.tests.fixtures import FakeProvider


class XcodeStructuredAgentTests(unittest.TestCase):
    def test_chat_turn_still_uses_normal_runtime_boundary(self) -> None:
        seen_tools: list[list[str]] = []

        def factory(messages, tools):
            seen_tools.append([tool.name for tool in tools])
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("<git-preflight>", messages[0]["content"])
            return [TextDelta("你好，我是 Xcode。"), FinalMessage("", "end_turn")]

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
        events: list[ProviderEvent] = [TextDelta("done"), FinalMessage("", "end_turn")]
        agent = StructuredAgent(
            provider=FakeProvider(events),
            registry=(),
        )

        result = agent.run("hello")

        self.assertEqual(result.answer, "done")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.tool_calls, [])
        assert result.metrics is not None
        self.assertEqual(result.metrics["llm_calls"], 1)

    def test_provider_events_drive_main_loop(self) -> None:
        events: list[ProviderEvent] = [TextDelta("done"), FinalMessage("", "end_turn")]
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
                        [
                            ToolCall("a", "echo", {"text": "one"}),
                            ToolCall("b", "echo", {"text": "two"}),
                        ]
                    ),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("finished"), FinalMessage("", "end_turn")],
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
                    ToolCallEvent([ToolCall("x", "missing", {})]),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("saw error"), FinalMessage("", "end_turn")],
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
            ToolCallEvent([ToolCall("x", "missing", {})]),
            FinalMessage("", "end_turn"),
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
                        ToolCallEvent([ToolCall("x", "echo", {"input": "same"})]),
                        FinalMessage("", "end_turn"),
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
                    ToolCallEvent([ToolCall("x", "echo", {"a": 1, "b": 2})]),
                    FinalMessage("", "end_turn"),
                ],
                [
                    ToolCallEvent([ToolCall("y", "echo", {"b": 2, "a": 1})]),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
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
                    [ToolCall(f"r{index}", "read_file", {"path": f"notes-{index}.md"})]
                ),
                FinalMessage("", "end_turn"),
            ]
            for index in range(4)
        ]
        last_event: list[ProviderEvent] = [
            TextDelta("done"),
            FinalMessage("", "end_turn"),
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
            return [TextDelta("done"), FinalMessage("", "end_turn")]

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
                ToolCallEvent([ToolCall("x", "edit_file", {"input": "hello"})]),
                FinalMessage("", "end_turn"),
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
                        [ToolCall("a", "bash", {"command": "git diff --stat"})]
                    ),
                    FinalMessage("", "end_turn"),
                ],
                [
                    ToolCallEvent(
                        [ToolCall("b", "bash", {"command": "python script.py"})]
                    ),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
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
                    ToolCallEvent([ToolCall("a", "echo", {"input": "hello"})]),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
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
        self.assertEqual(events[-1].data.answer, "done")

    def test_run_stream_yields_text_delta_events(self) -> None:
        mock_events: list[ProviderEvent] = [
            TextDelta("he"),
            TextDelta("llo"),
            FinalMessage("", "end_turn"),
        ]
        agent = StructuredAgent(
            provider=FakeProvider(mock_events),
            registry=(),
        )

        events = list(agent.run_stream("go"))

        self.assertEqual(
            [event.type for event in events],
            ["message_start", "text_delta", "text_delta", "assistant", "turn_end", "final"],
        )
        self.assertEqual(events[1].data, "he")
        self.assertEqual(events[-1].data.answer, "hello")

    def test_run_stream_uses_windows_selector_worker(self) -> None:
        if not hasattr(asyncio, "SelectorEventLoop"):
            self.skipTest("SelectorEventLoop is unavailable")

        mock_events: list[ProviderEvent] = [
            TextDelta("done"),
            FinalMessage("", "end_turn"),
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
        self.assertEqual(events[-1].data.answer, "done")

    def test_run_stream_does_not_call_asyncio_run_in_bridge(self) -> None:
        mock_events: list[ProviderEvent] = [
            TextDelta("done"),
            FinalMessage("", "end_turn"),
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

        self.assertEqual(events[-1].data.answer, "done")

    def test_arun_returns_result_inside_event_loop(self) -> None:
        async def main():
            events: list[ProviderEvent] = [
                TextDelta("async done"),
                FinalMessage("", "end_turn"),
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
                TextDelta("async done"),
                FinalMessage("", "end_turn"),
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
                TextDelta("he"),
                TextDelta("llo"),
                FinalMessage("", "end_turn"),
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
            ["message_start", "text_delta", "text_delta", "assistant", "turn_end", "final"],
        )
        self.assertEqual(events[-1].data.answer, "hello")

    def test_sync_api_rejects_active_event_loop(self) -> None:
        async def main():
            events: list[ProviderEvent] = [
                TextDelta("done"),
                FinalMessage("", "end_turn"),
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
                        [
                            ToolCall("a", "first", {}),
                            ToolCall("b", "second", {}),
                        ]
                    ),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
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
                    ToolCallEvent([ToolCall("x", "boom", {})]),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("recovered"), FinalMessage("", "end_turn")],
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
                ToolCallEvent([ToolCall("x", "echo", {})]),
                FinalMessage("", "end_turn"),
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
                return [TextDelta("part1"), FinalMessage("", "max_tokens")]
            else:
                return [TextDelta(" part2"), FinalMessage("", "end_turn")]

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
            return [TextDelta("x"), FinalMessage("", "max_tokens")]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            agent.run("hello")
        self.assertIn("Diminishing Returns", str(ctx.exception))

    def test_transient_error_retry(self) -> None:
        import unittest.mock as mock

        calls = []

        def factory(messages, tools) -> list[ProviderEvent]:
            calls.append(messages)
            if len(calls) < 3:
                raise RuntimeError("529 overloaded")
            return [TextDelta("success"), FinalMessage("", "end_turn")]

        agent = StructuredAgent(
            provider=FakeProvider(factory),
            registry=(),
        )
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()) as mock_sleep:
            result = agent.run("hello")
            self.assertEqual(result.answer, "success")
            self.assertEqual(len(calls), 3)
            self.assertEqual(mock_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
