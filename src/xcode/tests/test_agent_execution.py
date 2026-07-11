"""工具执行调度、看门狗、签名等纯函数单元测试。"""

from __future__ import annotations

from xcode.agent.config import _LoopRunState, AgentLoopConfig
from xcode.agent._execution import (
    partition_tool_calls_for_execution,
    tool_call_signature,
    tool_calls_signature,
    is_file_mutation_tool,
    is_file_read_tool,
    should_clear_read_history,
    is_tool_productive_default,
    update_repeated_tool_watchdog,
    update_idle_tool_watchdog,
)
from xcode.agent.messages import ToolResultMessage
from xcode.agent.types import (
    AgentTool,
    ToolCallContent,
)
from xcode.agent.config import AgentContext


# ── mock AgentTool ──


class _MockTool(AgentTool):
    def __init__(self, name: str, mode: str | None = None) -> None:
        self._name = name
        self._mode = mode

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
    def parameters(self) -> dict[str, object]:
        return {}

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
