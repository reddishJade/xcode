"""Agent 不可见的 watchdog 错误/成功结果协作 oracle。"""

from xcode.agent.config import AgentLoopConfig, _LoopRunState
from xcode.agent.messages import ToolResultMessage
from xcode.agent.types import ToolCallContent
from xcode.agent.watchdog import update_repeated_tool_watchdog


def _call() -> ToolCallContent:
    return ToolCallContent(
        id="call-1",
        name="write_file",
        arguments={"path": "target.txt", "content": "value"},
    )


def _result(*, error: bool) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="call-1",
        tool_name="write_file",
        content="disk full" if error else "ok",
        is_error=error,
    )


def test_all_error_repetitions_do_not_mask_idle_watchdog() -> None:
    state = _LoopRunState()
    config = AgentLoopConfig(watchdog_repeated_tool_limit=3)
    for _ in range(5):
        reason = update_repeated_tool_watchdog(
            state,
            [_call()],
            config,
            [_result(error=True)],
        )
        assert reason is None
        assert state.repeated_tool_count == 0


def test_successful_repetitions_remain_bounded() -> None:
    state = _LoopRunState()
    config = AgentLoopConfig(watchdog_repeated_tool_limit=3)
    reasons = [
        update_repeated_tool_watchdog(
            state,
            [_call()],
            config,
            [_result(error=False)],
        )
        for _ in range(4)
    ]
    assert reasons[:3] == [None, None, None]
    assert reasons[3] is not None
    assert "repeated tool call" in reasons[3]


def test_error_result_resets_prior_successful_repetition_count() -> None:
    state = _LoopRunState()
    config = AgentLoopConfig(watchdog_repeated_tool_limit=3)
    update_repeated_tool_watchdog(state, [_call()], config, [_result(error=False)])
    update_repeated_tool_watchdog(state, [_call()], config, [_result(error=False)])
    assert state.repeated_tool_count == 1

    update_repeated_tool_watchdog(state, [_call()], config, [_result(error=True)])

    assert state.repeated_tool_count == 0
    assert state.last_tool_signature is None
