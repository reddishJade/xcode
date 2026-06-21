from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import cast
from unittest.mock import patch

from xcode.agent.tool_execution import (
    ExecutedToolBatch,
    execute_tool_calls,
    partition_tool_calls_for_execution,
)
from xcode.agent.config import (
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from xcode.agent.events import AgentEvent, ToolExecutionEndEvent
from xcode.agent.messages import AssistantMessage
from xcode.agent.protocols import AgentTool
from xcode.agent.types import (
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ToolCallContent,
)
from xcode.harness.agent_runtime.tool_adapter import adapt_tool_specs
from xcode.harness.agent_runtime.cancellation import CancellationToken
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
from xcode.harness.observability import (
    HITLResult,
    PermissionEngine,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.skills import (
    AGENT_CONTENT_BLOCKS_METADATA_KEY,
    ToolOutput,
    ToolSpec,
)


EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class AgentToolExecutionTests(unittest.TestCase):
    def test_parallel_tools_respect_worker_limit(self) -> None:
        """parallel batch 的活跃 handler 数不超过 tool_workers。"""
        lock = threading.Lock()
        active = 0
        max_active = 0

        def handler(_data: dict[str, object]) -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return "ok"

        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "read",
                    "Read.",
                    "{}",
                    handler,
                    read_only=True,
                    concurrency_safe=True,
                    schema=EMPTY_SCHEMA,
                ),
            )
        )
        calls = [
            ToolCallContent(id=f"call-{index}", name="read", arguments={})
            for index in range(6)
        ]

        result = asyncio.run(
            execute_tool_calls(
                AgentContext(tools=cast(list[AgentTool], [tool])),
                AssistantMessage(content=list(calls)),
                calls,
                AgentLoopConfig(tool_workers=2),
                None,
                lambda _event: None,
            )
        )

        self.assertEqual(len(result.results), 6)
        self.assertEqual(max_active, 2)

    def test_cancelled_waiting_parallel_tool_does_not_start_handler(self) -> None:
        """等待并发额度的工具在取消后不进入 handler。"""
        started = threading.Event()
        release = threading.Event()
        calls_started = 0

        def handler(_data: dict[str, object]) -> str:
            nonlocal calls_started
            calls_started += 1
            started.set()
            release.wait(timeout=1)
            return "ok"

        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "read",
                    "Read.",
                    "{}",
                    handler,
                    read_only=True,
                    concurrency_safe=True,
                    schema=EMPTY_SCHEMA,
                ),
            )
        )
        calls = [
            ToolCallContent(id=f"call-{index}", name="read", arguments={})
            for index in range(2)
        ]
        token = CancellationToken()

        async def run_cancelled_batch() -> ExecutedToolBatch:
            task = asyncio.create_task(
                execute_tool_calls(
                    AgentContext(tools=cast(list[AgentTool], [tool])),
                    AssistantMessage(content=list(calls)),
                    calls,
                    AgentLoopConfig(tool_workers=1),
                    token,
                    lambda _event: None,
                )
            )
            await asyncio.to_thread(started.wait, 1)
            token.cancel("cancelled in test")
            release.set()
            return await task

        result = asyncio.run(run_cancelled_batch())

        self.assertEqual(calls_started, 1)
        self.assertEqual(len(result.results), 2)
        self.assertIn("cancelled in test", str(result.results[1].content))

    def test_parallel_worker_limit_collects_handler_errors(self) -> None:
        """受限并发仍收集每个工具结果，包括 handler 异常。"""

        def handler(data: dict[str, object]) -> str:
            if data.get("fail"):
                raise ValueError("expected failure")
            return "ok"

        schema = {
            "type": "object",
            "properties": {"fail": {"type": "boolean"}},
            "additionalProperties": False,
        }
        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "read",
                    "Read.",
                    "{}",
                    handler,
                    read_only=True,
                    concurrency_safe=True,
                    schema=schema,
                ),
            )
        )
        calls = [
            ToolCallContent(id="ok-1", name="read", arguments={}),
            ToolCallContent(id="bad", name="read", arguments={"fail": True}),
            ToolCallContent(id="ok-2", name="read", arguments={}),
        ]

        result = asyncio.run(
            execute_tool_calls(
                AgentContext(tools=cast(list[AgentTool], [tool])),
                AssistantMessage(content=list(calls)),
                calls,
                AgentLoopConfig(tool_workers=1),
                None,
                lambda _event: None,
            )
        )

        self.assertEqual(len(result.results), 3)
        self.assertFalse(result.results[0].is_error)
        self.assertTrue(result.results[1].is_error)
        self.assertFalse(result.results[2].is_error)

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
                    schema=EMPTY_SCHEMA,
                ),
                ToolSpec(
                    "write",
                    "Write.",
                    "text",
                    lambda _data: "",
                    schema=EMPTY_SCHEMA,
                ),
                ToolSpec(
                    "explicit",
                    "Explicit.",
                    "text",
                    lambda _data: "",
                    execution_mode="parallel",
                    schema=EMPTY_SCHEMA,
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
                    schema=EMPTY_SCHEMA,
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
            (
                ToolSpec(
                    "shell",
                    "Run shell.",
                    "{}",
                    lambda _data: output,
                    schema=EMPTY_SCHEMA,
                ),
            )
        )

        result = asyncio.run(tool.execute("call-1", {}))

        self.assertIsInstance(result.content[1], ShellCallOutputContent)
        block = result.content[1]
        assert isinstance(block, ShellCallOutputContent)
        self.assertEqual(block.call_id, "call-1")
        self.assertEqual(block.output[0]["stdout"], "ok")

    def test_toolspec_adapter_preserves_typed_result_and_details(self) -> None:
        """ToolOutput 的类型块、详情和错误状态会进入 AgentToolResult。"""
        image = ImageContent(
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "encoded",
            }
        )
        file = FileContent(filename="result.txt", file_data="content")
        output = ToolOutput(
            "summary",
            metadata={
                AGENT_CONTENT_BLOCKS_METADATA_KEY: [image, file],
                "provider_result": {"status": "invalid"},
            },
            is_error=True,
        )
        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "typed",
                    "Return typed content.",
                    "{}",
                    lambda _data: output,
                    schema=EMPTY_SCHEMA,
                ),
            )
        )

        result = asyncio.run(tool.execute("call-1", {}))

        self.assertTrue(result.is_error)
        self.assertEqual(result.details["provider_result"], {"status": "invalid"})
        self.assertIs(result.content[1], image)
        self.assertIs(result.content[2], file)

    def test_tool_execution_message_preserves_non_text_blocks(self) -> None:
        """工具执行消息不会在存在非文本块时只保留文本。"""
        image = ImageContent(source={"type": "base64", "data": "encoded"})
        output = ToolOutput(
            "summary",
            metadata={AGENT_CONTENT_BLOCKS_METADATA_KEY: [image]},
        )
        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "typed",
                    "Return typed content.",
                    "{}",
                    lambda _data: output,
                    schema=EMPTY_SCHEMA,
                ),
            )
        )
        tool_call = ToolCallContent(id="call-1", name="typed", arguments={})
        context = AgentContext(tools=cast(list[AgentTool], [tool]))

        batch = asyncio.run(
            execute_tool_calls(
                context,
                AssistantMessage(content=[tool_call]),
                [tool_call],
                AgentLoopConfig(),
                None,
                lambda _event: None,
            )
        )

        message_content = batch.results[0].content
        self.assertIsInstance(message_content, list)
        assert isinstance(message_content, list)
        self.assertIs(message_content[1], image)

    def test_toolspec_adapter_default_allow(self) -> None:
        """无风险审批后，工具默认 allow。"""
        called = False

        def handler(_data: dict) -> str:
            nonlocal called
            called = True
            return "changed"

        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "write",
                    "Write.",
                    "text",
                    handler,
                    schema=EMPTY_SCHEMA,
                ),
            )
        )

        result = asyncio.run(tool.execute("call-1", {}))
        self.assertTrue(called)
        self.assertFalse(result.is_error)

    def test_toolspec_adapter_execute_runs_handler_directly(self) -> None:
        """ToolSpecAdapter 直接执行 handler，不检查权限（权限由 ToolGate 门控）。"""
        called = False

        def handler(_data: dict) -> str:
            nonlocal called
            called = True
            return "changed"

        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "write",
                    "Write.",
                    "text",
                    handler,
                    schema=EMPTY_SCHEMA,
                ),
            ),
        )

        result = asyncio.run(tool.execute("call-1", {}))

        self.assertTrue(called)
        self.assertFalse(result.is_error)
        block = result.content[0]
        self.assertIsInstance(block, TextContent)
        assert isinstance(block, TextContent)
        self.assertEqual(block.text, "changed")

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
                    schema=EMPTY_SCHEMA,
                ),
                ToolSpec(
                    "write",
                    "Write.",
                    "text",
                    lambda _data: "",
                    schema=EMPTY_SCHEMA,
                ),
                ToolSpec(
                    "unsafe_read",
                    "Unsafe read.",
                    "text",
                    lambda _data: "",
                    read_only=True,
                    concurrency_safe=False,
                    schema=EMPTY_SCHEMA,
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
        events: list[AgentEvent] = []
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
        end_event = events[-1]
        self.assertIsInstance(end_event, ToolExecutionEndEvent)
        assert isinstance(end_event, ToolExecutionEndEvent)
        self.assertTrue(end_event.is_error)

    def test_before_tool_block_emits_end_event(self) -> None:
        events: list[AgentEvent] = []
        (tool,) = adapt_tool_specs(
            (
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda _data: "ok",
                    schema=EMPTY_SCHEMA,
                ),
            )
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
        end_event = events[-1]
        self.assertIsInstance(end_event, ToolExecutionEndEvent)
        assert isinstance(end_event, ToolExecutionEndEvent)
        self.assertTrue(end_event.is_error)


class TestPermissionSingleGate(unittest.TestCase):
    """验证 PermissionEngine.decide() 在完整的工具执行路径中恰好调用一次。

    ToolGate 是唯一的权限门控点。ToolSpecAdapter 不执行任何权限检查。
    所有测试通过 monkeypatch PermissionEngine.decide 计数来验证调用次数。
    """

    TOOL_NAME = "test_tool"

    def _handler_ok(self, _data: object) -> str:
        self._handler_called = True
        return "ok"

    def _handler_never(self, _data: object) -> str:
        self.fail("handler should not be called when tool is denied")

    def _make_spec(self, handler):
        return ToolSpec(
            self.TOOL_NAME,
            "test description",
            "{}",
            handler,
            schema=EMPTY_SCHEMA,
        )

    def _make_gate(self, policy: PermissionPolicy | None, callback=None) -> ToolGate:
        return ToolGate(
            mode_state=ExecutionModeState(),
            approval_callback=callback,
            permission_policy=policy,
            hook_manager=None,
            audit_logger=None,
            session_id="test",
        )

    def _run_execution(
        self, gate: ToolGate, spec: ToolSpec, args: dict | None = None
    ) -> ExecutedToolBatch:
        adapted = gate.adapt_tools((spec,))
        snapshot = gate.snapshot_for((spec,))
        config = AgentLoopConfig(
            before_tool_call=gate.build_before_tool_hook(snapshot),
        )
        tool_call = ToolCallContent(id="call-1", name=spec.name, arguments=args or {})
        return asyncio.run(
            execute_tool_calls(
                AgentContext(tools=cast(list[AgentTool], adapted)),
                AssistantMessage(content=[tool_call]),
                [tool_call],
                config,
                None,
                lambda _: None,
            )
        )

    # ── allow path ──

    def test_allow_path_calls_permission_once(self) -> None:
        """allow 路径：PermissionEngine.decide() 调用 1 次，handler 运行。"""
        self._handler_called = False
        spec = self._make_spec(self._handler_ok)
        gate = self._make_gate(
            PermissionPolicy((StaticPermission(self.TOOL_NAME, "allow"),))
        )

        decide_count: list[int] = [0]
        orig_decide = PermissionEngine.decide

        def counting_decide(self, tool_name, action_input, **kwargs):
            decide_count[0] += 1
            return orig_decide(self, tool_name, action_input, **kwargs)

        with patch.object(PermissionEngine, "decide", counting_decide):
            result = self._run_execution(gate, spec)

        self.assertEqual(decide_count[0], 1, msg="decide 必须恰好调用一次")
        self.assertTrue(self._handler_called, msg="handler 必须在 allow 时执行")
        self.assertFalse(result.results[0].is_error)

    # ── deny path ──

    def test_deny_path_calls_permission_once_handler_skipped(self) -> None:
        """deny 路径：PermissionEngine.decide() 调用 1 次，handler 不运行。"""
        spec = self._make_spec(self._handler_never)
        gate = self._make_gate(
            PermissionPolicy((StaticPermission(self.TOOL_NAME, "deny"),))
        )

        decide_count: list[int] = [0]
        orig_decide = PermissionEngine.decide

        def counting_decide(self, tool_name, action_input, **kwargs):
            decide_count[0] += 1
            return orig_decide(self, tool_name, action_input, **kwargs)

        with patch.object(PermissionEngine, "decide", counting_decide):
            result = self._run_execution(gate, spec)

        self.assertEqual(decide_count[0], 1, msg="decide 必须恰好调用一次")
        self.assertTrue(result.results[0].is_error, msg="deny 工具必须返回 error")

    # ── ask/defer path ──

    def test_ask_defer_path_calls_permission_once_blocked(self) -> None:
        """ask 路径：PermissionEngine.decide() 调用 1 次，工具被 block。

        ask blocks when no approval mechanism exists.
        handler 不执行；grant 在后续调用中满足。
        *ask/grant 的完整周期测试见 test_xcode_permissions.py。
        """
        spec = self._make_spec(self._handler_never)
        gate = self._make_gate(
            PermissionPolicy((StaticPermission(self.TOOL_NAME, "ask"),)),
            callback=lambda _t, _i: HITLResult("allow", "session"),
        )

        decide_count: list[int] = [0]
        orig_decide = PermissionEngine.decide

        def counting_decide(self, tool_name, action_input, **kwargs):
            decide_count[0] += 1
            return orig_decide(self, tool_name, action_input, **kwargs)

        with patch.object(PermissionEngine, "decide", counting_decide):
            result = self._run_execution(gate, spec)

        self.assertEqual(decide_count[0], 1, msg="decide 必须恰好调用一次")
        self.assertTrue(result.results[0].is_error, msg="ask defer 必须返回 error")

    # ── ToolSpecAdapter direct (no PermissionEngine) ──

    def test_adapter_direct_execute_no_permission(self) -> None:
        """ToolSpecAdapter.execute() 直接调用不涉及 PermissionEngine。"""
        called = False

        def handler(_data: dict) -> str:
            nonlocal called
            called = True
            return "direct"

        (tool,) = adapt_tool_specs(
            (ToolSpec("direct", "test", "{}", handler, schema=EMPTY_SCHEMA),)
        )
        result = asyncio.run(tool.execute("call-1", {}))
        self.assertTrue(called)
        self.assertFalse(result.is_error)
        self.assertIn("direct", str(result.content))

    def test_adapter_has_no_engine_attribute(self) -> None:
        """ToolSpecAdapter 实例不包含 _engine 属性（PermissionEngine 已剥离）。"""
        (tool,) = adapt_tool_specs(
            (ToolSpec("x", "test", "{}", lambda _: "", schema=EMPTY_SCHEMA),)
        )
        self.assertFalse(hasattr(tool, "_engine"), msg="adapter 不应持有 _engine")

    # ── production execution routes through ToolGate ──

    def test_production_flow_routes_through_toolgate(self) -> None:
        """验证完整生产路径经过 ToolGate 门控（通过 execute_tool_calls）。"""
        self._handler_called = False
        spec = self._make_spec(self._handler_ok)
        gate = self._make_gate(
            PermissionPolicy((StaticPermission(self.TOOL_NAME, "allow"),))
        )

        decide_count: list[int] = [0]
        orig_decide = PermissionEngine.decide

        def counting_decide(self, tool_name, action_input, **kwargs):
            decide_count[0] += 1
            return orig_decide(self, tool_name, action_input, **kwargs)

        with patch.object(PermissionEngine, "decide", counting_decide):
            result = self._run_execution(gate, spec)

        self.assertEqual(decide_count[0], 1, msg="生产路径必须经过 ToolGate")
        self.assertTrue(self._handler_called)
        self.assertFalse(result.results[0].is_error)


if __name__ == "__main__":
    unittest.main()
