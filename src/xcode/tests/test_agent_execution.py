"""工具执行调度、看门狗、签名等纯函数单元测试。"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from xcode.agent._execution import (
    _run_tool_handler,
    execute_tool_calls,
    is_file_mutation_tool,
    is_file_read_tool,
    is_tool_productive_default,
    partition_tool_calls_for_execution,
    should_clear_read_history,
    tool_call_signature,
    tool_calls_signature,
    update_idle_tool_watchdog,
    update_repeated_tool_watchdog,
    validate_tool_arguments,
)
from xcode.agent.config import AgentContext, AgentLoopConfig, _LoopRunState
from xcode.agent.messages import AssistantMessage, ToolResultMessage
from xcode.agent.types import (
    AgentTool,
    AgentToolResult,
    TextContent,
    ToolCallContent,
    ToolSpecAdapter,
)
from xcode.coding_agent.tools.bash import build_bash_tool
from xcode.harness.agent_runtime.cancellation import CancellationToken

# ── mock AgentTool ──


class _MockTool(AgentTool):
    def __init__(
        self,
        name: str,
        mode: str | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        self._name = name
        self._mode = mode
        self._parameters = parameters or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return ""

    @property
    def parameters(self) -> Mapping[str, object]:
        return self._parameters

    @property
    def execution_mode(self) -> str | None:
        return self._mode

    @property
    def examples(self) -> list[dict[str, object]]:
        return []

    async def execute(self, *args, **kwargs) -> object:
        from xcode.agent.types import AgentToolResult, TextContent

        return AgentToolResult(content=[TextContent(text="ok")])


class TestPartitionToolCalls:
    def test_single_batch_all_parallel(self) -> None:
        ctx = AgentContext(tools=[_MockTool("read", "parallel")])
        calls = [
            ToolCallContent(id="1", name="read"),
            ToolCallContent(id="2", name="read"),
        ]
        batches = partition_tool_calls_for_execution(ctx, calls)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_sequential_tools_separate_batches(self) -> None:
        ctx = AgentContext(tools=[_MockTool("write", "sequential")])
        calls = [
            ToolCallContent(id="1", name="write"),
            ToolCallContent(id="2", name="write"),
        ]
        batches = partition_tool_calls_for_execution(ctx, calls)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    def test_mixed_modes(self) -> None:
        ctx = AgentContext(
            tools=[_MockTool("read", "parallel"), _MockTool("write", "sequential")]
        )
        calls = [
            ToolCallContent(id="1", name="read"),
            ToolCallContent(id="2", name="read"),
            ToolCallContent(id="3", name="write"),
        ]
        batches = partition_tool_calls_for_execution(ctx, calls)
        assert len(batches) == 2
        assert len(batches[0]) == 2  # parallel batch
        assert batches[1][0].id == "3"  # sequential batch

    def test_unknown_tool_defaults_to_sequential(self) -> None:
        ctx = AgentContext(tools=[])
        calls = [ToolCallContent(id="1", name="unknown")]
        batches = partition_tool_calls_for_execution(ctx, calls)
        assert len(batches) == 1
        assert len(batches[0]) == 1


def test_validate_tool_arguments_materializes_frozen_schema() -> None:
    properties: Mapping[str, object] = MappingProxyType(
        {"command": MappingProxyType({"type": "string"})}
    )
    schema: Mapping[str, object] = MappingProxyType(
        {
            "type": "object",
            "properties": properties,
            "required": ("command",),
            "additionalProperties": False,
        }
    )
    tool = _MockTool("bash", parameters=schema)
    call = ToolCallContent(id="1", name="bash", arguments={"command": "pwd"})

    assert validate_tool_arguments(tool, call, {"command": "pwd"}) is None


def test_validate_tool_arguments_reports_invalid_schema() -> None:
    tool = _MockTool("broken", parameters={"type": "not-a-json-schema-type"})
    call = ToolCallContent(id="1", name="broken", arguments={})

    error = validate_tool_arguments(tool, call, {})

    assert error is not None
    assert error.startswith("tool schema error for broken:")


class TestToolCallSignature:
    def test_consistency(self) -> None:
        c1 = ToolCallContent(id="a", name="read", arguments={"path": "/x"})
        c2 = ToolCallContent(id="b", name="read", arguments={"path": "/x"})
        assert tool_call_signature(c1) == tool_call_signature(c2)

    def test_different_args_different_signature(self) -> None:
        c1 = ToolCallContent(id="a", name="read", arguments={"path": "/x"})
        c2 = ToolCallContent(id="b", name="read", arguments={"path": "/y"})
        assert tool_call_signature(c1) != tool_call_signature(c2)


class TestToolCallsSignature:
    def test_sorted_parts(self) -> None:
        calls = [
            ToolCallContent(id="1", name="b", arguments={}),
            ToolCallContent(id="2", name="a", arguments={}),
        ]
        sig = tool_calls_signature(calls)
        assert "a:" in sig[:4]
        # sorted order: a before b


class TestFileToolClassification:
    def test_is_mutation_tool(self) -> None:
        assert is_file_mutation_tool("write_file")
        assert not is_file_mutation_tool("read_file")

    def test_is_read_tool(self) -> None:
        assert is_file_read_tool("read_file")
        assert not is_file_read_tool("write_file")

    def test_custom_sets(self) -> None:
        custom_mutation = frozenset({"my_write"})
        assert is_file_mutation_tool("my_write", custom_mutation)
        assert not is_file_mutation_tool("write_file", custom_mutation)


class TestShouldClearReadHistory:
    def test_mutation_clears(self) -> None:
        calls = [ToolCallContent(id="1", name="write_file")]
        assert should_clear_read_history(calls, [])

    def test_read_only_does_not_clear(self) -> None:
        calls = [ToolCallContent(id="1", name="read_file")]
        assert not should_clear_read_history(calls, [])


class TestIsToolProductiveDefault:
    def test_all_ok(self) -> None:
        results = [
            ToolResultMessage(
                tool_call_id="c1", tool_name="t", content="ok", is_error=False
            ),
        ]
        assert is_tool_productive_default([], results)

    def test_all_error(self) -> None:
        results = [
            ToolResultMessage(
                tool_call_id="c1", tool_name="t", content="err", is_error=True
            ),
        ]
        assert not is_tool_productive_default([], results)


def _make_call(name: str, **kwargs: str) -> ToolCallContent:
    return ToolCallContent(id=kwargs.get("id", "c1"), name=name, arguments={"k": "v"})


class TestUpdateRepeatedToolWatchdog:
    def test_no_repeat_no_trigger(self) -> None:
        state = _LoopRunState()
        config = AgentLoopConfig()
        calls = [_make_call("read", id="c1")]
        results = [ToolResultMessage(tool_call_id="c1", tool_name="read", content="ok")]
        assert update_repeated_tool_watchdog(state, calls, config, results) is None
        assert state.repeated_tool_count == 0

    def test_repeat_triggers_watchdog(self) -> None:
        state = _LoopRunState()
        config = AgentLoopConfig(watchdog_repeated_tool_limit=3)
        calls = [_make_call("read", id="c1")]
        results = [ToolResultMessage(tool_call_id="c1", tool_name="read", content="ok")]

        reason = None
        for _ in range(4):  # first call doesn't match, next 3 do
            reason = update_repeated_tool_watchdog(state, calls, config, results)
        assert reason is not None
        assert "watchdog" in reason

    def test_skipped_tools_not_counted(self) -> None:
        state = _LoopRunState()
        config = AgentLoopConfig(
            watchdog_repeated_tool_limit=2,
            watchdog_repeated_tool_skip=frozenset({"skip_tool"}),
        )
        skip_call = [_make_call("skip_tool", id="c1")]
        results = [
            ToolResultMessage(tool_call_id="c1", tool_name="skip_tool", content="ok")
        ]

        reason = update_repeated_tool_watchdog(state, skip_call, config, results)
        assert reason is None

    def test_error_results_reset_counter(self) -> None:
        state = _LoopRunState()
        config = AgentLoopConfig()
        calls = [_make_call("read", id="c1")]
        results = [
            ToolResultMessage(
                tool_call_id="c1", tool_name="read", content="err", is_error=True
            )
        ]
        update_repeated_tool_watchdog(state, calls, config, results)
        assert state.repeated_tool_count == 0


class TestUpdateIdleToolWatchdog:
    def test_productive_resets_counter(self) -> None:
        state = _LoopRunState(consecutive_idle_steps=3)
        config = AgentLoopConfig(max_consecutive_idle_steps=4)
        calls = [_make_call("read")]
        results = [ToolResultMessage(tool_call_id="c1", tool_name="read", content="ok")]
        reason = update_idle_tool_watchdog(state, calls, results, config)
        assert reason is None
        assert state.consecutive_idle_steps == 0

    def test_unproductive_triggers_watchdog(self) -> None:
        state = _LoopRunState()
        config = AgentLoopConfig(max_consecutive_idle_steps=3)
        calls = [_make_call("read")]
        results = [
            ToolResultMessage(
                tool_call_id="c1", tool_name="read", content="err", is_error=True
            )
        ]

        for _ in range(3):
            reason = update_idle_tool_watchdog(state, calls, results, config)
        assert reason is not None
        assert "Watchdog" in reason


# ── 工具执行中途打断 ──


class _BlockingTool(AgentTool):
    """忽略协作式 token，直到外部释放或 task 被强制取消。"""

    def __init__(
        self,
        name: str,
        mode: str,
        release: threading.Event,
    ) -> None:
        self._name = name
        self._mode = mode
        self._release = release

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return ""

    @property
    def parameters(self) -> Mapping[str, object]:
        return {}

    @property
    def execution_mode(self) -> str:
        return self._mode

    @property
    def examples(self) -> list[dict[str, object]]:
        return []

    async def execute(self, *args: object, **kwargs: object) -> AgentToolResult:
        while not self._release.is_set():
            await asyncio.sleep(0.01)
        return AgentToolResult(content=[TextContent(text="blocked done")])


class _CancellationAwareTool(AgentTool):
    """token 取消后短暂延迟即返回真实结果，模拟 bash 自杀式取消。"""

    def __init__(self, token: CancellationToken) -> None:
        self._token = token

    @property
    def name(self) -> str:
        return "aware"

    @property
    def label(self) -> str:
        return "aware"

    @property
    def description(self) -> str:
        return ""

    @property
    def parameters(self) -> Mapping[str, object]:
        return {}

    @property
    def execution_mode(self) -> str | None:
        return None

    @property
    def examples(self) -> list[dict[str, object]]:
        return []

    async def execute(self, *args: object, **kwargs: object) -> AgentToolResult:
        while not self._token.is_cancelled():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        return AgentToolResult(content=[TextContent(text="Command cancelled")])


async def test_run_tool_handler_abandons_non_cancellable_tool(
    monkeypatch: object,
) -> None:
    """不可取消的工具在宽限期后放弃等待并返回打断结果。"""
    monkeypatch.setattr("xcode.agent._execution._TOOL_CANCEL_GRACE_SECONDS", 0.05)
    release = threading.Event()
    token = CancellationToken()
    tool = _BlockingTool("blocker", "parallel", release)
    call = ToolCallContent(id="b1", name="blocker")

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        token.cancel("interrupted by user")

    canceller = asyncio.create_task(cancel_soon())
    try:
        result, content, is_error, terminate = await _run_tool_handler(
            tool, call, {}, token, lambda _r: None, None
        )
        await canceller
        assert is_error
        assert not terminate
        assert (
            "".join(str(item.text) for item in content if isinstance(item, TextContent))
            == "interrupted by user"
        )
        assert result.is_error
    finally:
        release.set()


async def test_run_tool_handler_keeps_result_of_cancellable_tool() -> None:
    """可取消工具在宽限期内自行收尾时，保留其真实输出。"""
    token = CancellationToken()
    tool = _CancellationAwareTool(token)
    call = ToolCallContent(id="a1", name="aware")

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        token.cancel("interrupted by user")

    canceller = asyncio.create_task(cancel_soon())
    _result, content, is_error, terminate = await _run_tool_handler(
        tool, call, {}, token, lambda _r: None, None
    )
    await canceller

    assert not is_error
    assert not terminate
    text = "".join(str(item.text) for item in content if isinstance(item, TextContent))
    assert text == "Command cancelled"


async def test_parallel_batch_interrupt_returns_promptly(
    monkeypatch: object,
) -> None:
    """并行批次打断：阻塞工具被放弃，批次立即返回打断结果。"""
    monkeypatch.setattr("xcode.agent._execution._TOOL_CANCEL_GRACE_SECONDS", 0.05)
    release = threading.Event()
    token = CancellationToken()
    blocker = _BlockingTool("blocker", "parallel", release)
    ctx = AgentContext(tools=[blocker])
    call = ToolCallContent(id="b1", name="blocker", arguments={})
    message = AssistantMessage(content=[call])

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        token.cancel("interrupted by user")

    canceller = asyncio.create_task(cancel_soon())
    try:
        started = time.monotonic()
        batch = await execute_tool_calls(
            ctx,
            message,
            [call],
            AgentLoopConfig(tool_execution="parallel"),
            token,
            lambda _event: None,
        )
        elapsed = time.monotonic() - started
        await canceller
        assert elapsed < 2.0
        assert len(batch.results) == 1
        assert batch.results[0].is_error
        assert "interrupted by user" in str(batch.results[0].content)
        assert not batch.terminate
    finally:
        release.set()


async def test_interrupt_kills_long_running_bash(tmp_path: Path) -> None:
    """真实 bash：打断后 sleep 子进程被杀，工具返回取消输出。"""
    token = CancellationToken()
    bash_tool = build_bash_tool(tmp_path, cancel_event=token)
    ctx = AgentContext(tools=[ToolSpecAdapter(bash_tool)])
    call = ToolCallContent(
        id="b1",
        name="bash",
        arguments={"command": "sleep 30", "timeout_ms": 120_000},
    )
    message = AssistantMessage(content=[call])

    async def cancel_soon() -> None:
        await asyncio.sleep(0.3)
        token.cancel("interrupted by user")

    canceller = asyncio.create_task(cancel_soon())
    started = time.monotonic()
    batch = await execute_tool_calls(
        ctx,
        message,
        [call],
        AgentLoopConfig(tool_execution="parallel"),
        token,
        lambda _event: None,
    )
    elapsed = time.monotonic() - started
    await canceller

    assert elapsed < 2.0
    assert len(batch.results) == 1
    assert "cancelled" in str(batch.results[0].content).lower()
