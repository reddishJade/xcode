from __future__ import annotations

import asyncio
import unittest
from typing import cast

from xcode.agent.tool_execution import (
    execute_tool_calls,
    partition_tool_calls_for_execution,
)
from xcode.agent.config import (
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from xcode.agent.messages import AssistantMessage
from xcode.agent.protocols import AgentTool
from xcode.agent.types import ShellCallOutputContent, ToolCallContent
from xcode.harness.observability import HITLResult
from xcode.harness.agent_runtime.tool_adapter import adapt_tool_specs
from xcode.harness.skills import (
    AGENT_CONTENT_BLOCKS_METADATA_KEY,
    ToolOutput,
    ToolSpec,
)


class AgentToolExecutionTests(unittest.TestCase):
    def test_toolspec_adapter_derives_execution_mode_from_metadata(self) -> None:
        read_tool, write_tool, explicit_parallel = adapt_tool_specs(
            (
                ToolSpec(
                    "read",
                    "Read.",
                    "text",
                    lambda _data: "",
                    read_only=True,
                    concurrency_safe=True,
                ),
                ToolSpec("write", "Write.", "text", lambda _data: "", risk="high"),
                ToolSpec(
                    "explicit",
                    "Explicit.",
                    "text",
                    lambda _data: "",
                    execution_mode="parallel",
                ),
            )
        )

        self.assertEqual(read_tool.execution_mode, "parallel")
        self.assertEqual(write_tool.execution_mode, "sequential")
        self.assertEqual(explicit_parallel.execution_mode, "parallel")

    def test_toolspec_adapter_preserves_builtin_metadata(self) -> None:
        """ToolSpec builtin 元数据会传递给 AgentTool。"""
        builtin = {"type": "shell", "environment": {"type": "local"}}
        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "shell",
                    "Run shell.",
                    "{}",
                    lambda _data: "",
                    builtin=builtin,
                ),
            )
        )

        self.assertEqual(tool.builtin, builtin)

    def test_toolspec_adapter_preserves_shell_output_content(self) -> None:
        """ToolOutput 元数据中的 shell 输出块会保留给 agent loop。"""
        output = ToolOutput(
            "summary",
            metadata={
                AGENT_CONTENT_BLOCKS_METADATA_KEY: [
                    ShellCallOutputContent(
                        output=[
                            {
                                "stdout": "ok",
                                "stderr": "",
                                "outcome": {"type": "exit", "exit_code": 0},
                            }
                        ]
                    )
                ]
            },
        )
        (tool,) = adapt_tool_specs(
            (ToolSpec("shell", "Run shell.", "{}", lambda _data: output),)
        )

        result = asyncio.run(tool.execute("call-1", {}))

        self.assertIsInstance(result.content[1], ShellCallOutputContent)
        block = result.content[1]
        assert isinstance(block, ShellCallOutputContent)
        self.assertEqual(block.call_id, "call-1")
        self.assertEqual(block.output[0]["stdout"], "ok")

    def test_toolspec_adapter_blocks_high_risk_without_approval(self) -> None:
        called = False

        def handler(_data: dict) -> str:
            nonlocal called
            called = True
            return "changed"

        (tool,) = adapt_tool_specs(
            (ToolSpec("write", "Write.", "text", handler, risk="high"),)
        )

        result = asyncio.run(tool.execute("call-1", {}))

        self.assertFalse(called)
        self.assertTrue(result.is_error)
        self.assertIn("requires approval", result.content[0].text)

    def test_toolspec_adapter_runs_high_risk_after_approval(self) -> None:
        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "write", "Write.", "text", lambda _data: "changed", risk="high"
                ),
            ),
            approval_callback=lambda _tool, _input: HITLResult("allow", "once"),
        )

        result = asyncio.run(tool.execute("call-1", {}))

        self.assertFalse(result.is_error)
        self.assertEqual(result.content[0].text, "changed")

    def test_partition_tool_calls_for_execution_keeps_sequential_barriers(self) -> None:
        tools = adapt_tool_specs(
            (
                ToolSpec(
                    "read",
                    "Read.",
                    "text",
                    lambda _data: "",
                    read_only=True,
                    concurrency_safe=True,
                ),
                ToolSpec("write", "Write.", "text", lambda _data: "", risk="high"),
                ToolSpec(
                    "unsafe_read",
                    "Unsafe read.",
                    "text",
                    lambda _data: "",
                    read_only=True,
                    concurrency_safe=False,
                ),
            )
        )
        context = AgentContext(tools=cast(list[AgentTool], tools))
        tool_calls = [
            ToolCallContent(id="c1", name="read"),
            ToolCallContent(id="c2", name="read"),
            ToolCallContent(id="c3", name="write"),
            ToolCallContent(id="c4", name="read"),
            ToolCallContent(id="c5", name="unsafe_read"),
        ]

        batches = partition_tool_calls_for_execution(context, tool_calls)

        self.assertEqual(
            [[tool_call.id for tool_call in batch] for batch in batches],
            [["c1", "c2"], ["c3"], ["c4"], ["c5"]],
        )

    def test_unknown_tool_emits_end_event(self) -> None:
        events = []
        tool_call = ToolCallContent(id="missing-1", name="missing")

        result = asyncio.run(
            execute_tool_calls(
                AgentContext(),
                AssistantMessage(content=[tool_call]),
                [tool_call],
                AgentLoopConfig(),
                None,
                events.append,
            )
        )

        self.assertTrue(result.results[0].is_error)
        self.assertEqual(
            [event.type for event in events],
            ["tool_execution_start", "tool_execution_end"],
        )
        self.assertTrue(events[-1].is_error)

    def test_before_tool_block_emits_end_event(self) -> None:
        events = []
        (tool,) = adapt_tool_specs(
            (ToolSpec("echo", "Echo.", "text", lambda _data: "ok"),)
        )
        tool_call = ToolCallContent(id="echo-1", name="echo")

        def block_tool(
            _ctx: BeforeToolCallContext,
            _signal: object,
        ) -> BeforeToolCallResult:
            return BeforeToolCallResult(block=True, reason="blocked")

        result = asyncio.run(
            execute_tool_calls(
                AgentContext(tools=cast(list[AgentTool], [tool])),
                AssistantMessage(content=[tool_call]),
                [tool_call],
                AgentLoopConfig(before_tool_call=block_tool),
                None,
                events.append,
            )
        )

        self.assertTrue(result.results[0].is_error)
        self.assertEqual(result.results[0].content, "blocked")
        self.assertEqual(
            [event.type for event in events],
            ["tool_execution_start", "tool_execution_end"],
        )
        self.assertTrue(events[-1].is_error)


if __name__ == "__main__":
    unittest.main()
