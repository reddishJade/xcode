"""Agent 不可见的并行工具 worker 上限行为 oracle。"""

import asyncio
from unittest.mock import patch

from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.messages import AssistantMessage, ToolResultMessage
from xcode.agent.tool_execution import _execute_parallel
from xcode.agent.types import ToolCallContent
from xcode.harness.agent_runtime.config import build_loop_config, build_turn_snapshot
from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.config import AgentConfig, RequestHygieneConfig
from xcode.tests.fixtures import FakeProvider


def test_parallel_batch_never_exceeds_configured_worker_limit() -> None:
    active = 0
    maximum = 0

    async def execute_one(
        _context: AgentContext,
        _message: AssistantMessage,
        tool_call: ToolCallContent,
        _config: AgentLoopConfig,
        _signal: object,
        _emit: object,
    ) -> tuple[ToolResultMessage, bool]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.03)
        active -= 1
        return (
            ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content="ok",
            ),
            False,
        )

    calls = [
        ToolCallContent(id=f"call-{index}", name="read", arguments={})
        for index in range(6)
    ]

    with patch("xcode.agent.tool_execution._execute_one", side_effect=execute_one):
        result = asyncio.run(
            _execute_parallel(
                AgentContext(),
                AssistantMessage(content=list(calls)),
                calls,
                AgentLoopConfig(tool_workers=2),
                None,
                lambda _event: None,
            )
        )

    assert len(result.results) == 6
    assert maximum == 2


def test_public_agent_worker_setting_reaches_core_loop() -> None:
    provider = FakeProvider([])
    snapshot = build_turn_snapshot(
        AgentConfig(tool_workers=2),
        (),
        provider,
        None,
    )
    mode_state = ExecutionModeState()
    gate = ToolGate(mode_state, None, None, None, None, "hidden-test")

    loop = build_loop_config(
        "act",
        snapshot,
        gate,
        (),
        None,
        None,
        RequestHygieneConfig(enabled=False),
        None,
        None,
        lambda _registry, _mode: [],
        lambda _message: None,
        lambda _record: None,
        mode_state,
        lambda: "hidden-test",
    )

    assert loop.tool_workers == 2
