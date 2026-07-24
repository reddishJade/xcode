"""Goal 停止条件验证测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.messages import AgentMessage, UserMessage
from xcode.agent.results import TerminationReason
from xcode.ai.events import FinalMessage, ProviderEvent, TextDelta
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime.goal import GoalController, _parse_verdict


class _Provider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, object]]] = []

    @property
    def model(self) -> str:
        return "test-model"

    @property
    def base_url(self) -> str:
        return ""

    @property
    def transport(self) -> str:
        return "test"

    @property
    def thinking(self) -> bool:
        return False

    @property
    def reasoning_effort(self) -> str | None:
        return None

    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        del tools, options, kwargs
        self.requests.append(messages)
        response = self.responses.pop(0)
        yield TextDelta(response)
        yield FinalMessage(content="", stop_reason="end_turn")


def test_parse_verdict_accepts_json_fence() -> None:
    verdict = _parse_verdict(
        '```json\n{"ok": false, "reason": "tests were not run"}\n```'
    )

    assert not verdict.ok
    assert verdict.reason == "tests were not run"


@pytest.mark.asyncio
async def test_goal_rejects_then_accepts_from_transcript_evidence() -> None:
    provider = _Provider(
        [
            '{"ok": false, "reason": "missing test output"}',
            '{"ok": true, "reason": "pytest passed"}',
        ]
    )
    goal = GoalController(provider)
    goal.set("All tests pass")

    feedback = await goal.verify([UserMessage(content="please finish")])
    accepted = await goal.verify([UserMessage(content="pytest: 10 passed")])

    assert "missing test output" in str(feedback)
    assert accepted is None
    assert goal.condition is None
    assert "All tests pass" in str(provider.requests[0][-1]["content"])


@pytest.mark.asyncio
async def test_goal_reentry_is_bounded() -> None:
    provider = _Provider(
        ['{"ok": false, "reason": "still missing"}'] * 4,
    )
    goal = GoalController(provider, max_reacts=3)
    goal.set("Commit exists")

    feedback = [
        await goal.verify([UserMessage(content="working")]) for _ in range(4)
    ]

    assert all(feedback[index] is not None for index in range(3))
    assert feedback[3] is None
    assert goal.condition is None
    assert "3 re-entries" in str(goal.consume_terminal_notice())


@pytest.mark.asyncio
async def test_agent_loop_uses_completion_feedback_before_stopping() -> None:
    provider = _Provider(["premature answer", "finished after feedback"])
    verifier_calls = 0

    async def verifier(messages: list[AgentMessage]) -> str | None:
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 1:
            return "Run the tests before stopping."
        return None

    result = await run_agent_loop(
        [UserMessage(content="finish the task")],
        AgentContext(),
        AgentLoopConfig(
            provider=provider,
            max_steps=3,
            completion_verifier=verifier,
        ),
        lambda _event: None,
    )

    assert verifier_calls == 2
    assert any(
        isinstance(message, UserMessage)
        and "Run the tests" in str(message.content)
        for message in result.messages
    )
    assert result.steps == 2


@pytest.mark.asyncio
async def test_rejected_goal_at_step_limit_does_not_report_completed() -> None:
    provider = _Provider(["premature answer"])

    async def reject(_messages: list[AgentMessage]) -> str | None:
        return "Missing verification."

    result = await run_agent_loop(
        [UserMessage(content="finish the task")],
        AgentContext(),
        AgentLoopConfig(
            provider=provider,
            max_steps=1,
            completion_verifier=reject,
        ),
        lambda _event: None,
    )

    assert result.termination_reason is TerminationReason.STEP_LIMIT
