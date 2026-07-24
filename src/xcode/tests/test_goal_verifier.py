"""Goal 停止条件验证测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.messages import UserMessage
from xcode.agent.results import TerminationReason
from xcode.ai.events import FinalMessage, ProviderEvent, TextDelta
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime.goal import (
    GoalController,
    GoalDecision,
    GoalDecisionStatus,
    GoalVerdict,
    _parse_verdict,
)


class _Provider:
    def __init__(self, responses: list[str | FinalMessage]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, object]]] = []
        self.request_tools: list[list[ToolDefinition]] = []
        self.request_options: list[StreamOptions | None] = []

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
        del kwargs
        self.requests.append(messages)
        self.request_tools.append(tools)
        self.request_options.append(options)
        response = self.responses.pop(0)
        if isinstance(response, FinalMessage):
            yield response
            return
        yield TextDelta(response)
        yield FinalMessage(content="", stop_reason="end_turn")


def test_parse_verdict_accepts_json_fence() -> None:
    verdict = _parse_verdict(
        '```json\n{"ok": false, "reason": "tests were not run"}\n```'
    )

    assert verdict == GoalVerdict(
        ok=False,
        reason="tests were not run",
        impossible=False,
    )


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("not json", "judge returned no JSON verdict"),
        ('{"reason": "missing ok"}', "judge verdict must contain boolean ok"),
        ('{"ok": true}', "judge verdict must contain reason"),
    ],
)
def test_parse_verdict_rejects_invalid_contract(
    response: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _parse_verdict(response)


@pytest.mark.asyncio
async def test_verify_without_goal_skips_judge() -> None:
    provider = _Provider([])
    goal = GoalController(provider)

    decision = await goal.verify([UserMessage(content="done")])

    assert decision == GoalDecision(status=GoalDecisionStatus.NO_GOAL)
    assert provider.requests == []


@pytest.mark.asyncio
async def test_goal_returns_typed_rejection_then_acceptance() -> None:
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

    assert feedback == GoalDecision(
        status=GoalDecisionStatus.UNSATISFIED,
        reason="missing test output",
        feedback=(
            "<goal-verification>\n"
            "Stop condition is not yet satisfied: missing test output\n"
            "Continue working and produce transcript evidence before stopping.\n"
            "</goal-verification>"
        ),
    )
    assert accepted == GoalDecision(
        status=GoalDecisionStatus.SATISFIED,
        reason="pytest passed",
    )
    assert goal.condition is None
    assert provider.requests[0][1] == {
        "role": "user",
        "content": "please finish",
    }
    assert provider.requests[0][-1] == {
        "role": "user",
        "content": (
            "Has this stop condition been satisfied?\n\nCondition: All tests pass"
        ),
    }
    assert provider.request_tools == [[], []]
    assert provider.request_options == [
        StreamOptions(temperature=0, max_tokens=512),
        StreamOptions(temperature=0, max_tokens=512),
    ]


@pytest.mark.asyncio
async def test_goal_reentry_is_bounded() -> None:
    provider = _Provider(
        ['{"ok": false, "reason": "still missing"}'] * 4,
    )
    goal = GoalController(provider, max_reacts=3)
    goal.set("Commit exists")

    decisions = [await goal.verify([UserMessage(content="working")]) for _ in range(4)]

    assert [decision.status for decision in decisions] == [
        GoalDecisionStatus.UNSATISFIED,
        GoalDecisionStatus.UNSATISFIED,
        GoalDecisionStatus.UNSATISFIED,
        GoalDecisionStatus.REENTRY_LIMIT,
    ]
    assert [decision.reason for decision in decisions] == ["still missing"] * 4
    assert len(provider.requests) == 4
    assert goal.condition is None
    assert goal.consume_terminal_notice() == (
        "Goal remains unsatisfied after 3 re-entries: still missing"
    )


@pytest.mark.asyncio
async def test_agent_loop_reenters_through_real_goal_controller() -> None:
    provider = _Provider(
        [
            "premature answer",
            '{"ok": false, "reason": "tests were not run"}',
            "finished after feedback",
            '{"ok": true, "reason": "pytest passed"}',
        ]
    )
    goal = GoalController(provider)
    goal.set("Run the tests")

    result = await run_agent_loop(
        [UserMessage(content="finish the task")],
        AgentContext(),
        AgentLoopConfig(
            provider=provider,
            max_steps=3,
            completion_verifier=goal.completion_feedback,
        ),
        lambda _event: None,
    )

    feedback_messages = [
        message.content
        for message in result.messages
        if isinstance(message, UserMessage)
        and str(message.content).startswith("<goal-verification>")
    ]
    assert feedback_messages == [
        (
            "<goal-verification>\n"
            "Stop condition is not yet satisfied: tests were not run\n"
            "Continue working and produce transcript evidence before stopping.\n"
            "</goal-verification>"
        )
    ]
    assert result.steps == 2
    assert result.termination_reason is TerminationReason.COMPLETED
    assert len(provider.requests) == 4
    assert goal.condition is None


@pytest.mark.asyncio
async def test_rejected_goal_at_step_limit_does_not_report_completed() -> None:
    provider = _Provider(
        [
            "premature answer",
            '{"ok": false, "reason": "missing verification"}',
        ]
    )
    goal = GoalController(provider)
    goal.set("Verification exists")

    result = await run_agent_loop(
        [UserMessage(content="finish the task")],
        AgentContext(),
        AgentLoopConfig(
            provider=provider,
            max_steps=1,
            completion_verifier=goal.completion_feedback,
        ),
        lambda _event: None,
    )

    assert result.termination_reason is TerminationReason.STEP_LIMIT
    assert result.steps == 1
    assert len(provider.requests) == 2
    assert goal.condition == "Verification exists"


@pytest.mark.asyncio
async def test_impossible_goal_returns_typed_decision_and_clears_goal() -> None:
    provider = _Provider(
        [
            (
                '{"ok": false, "impossible": true, '
                '"reason": "required service is unavailable"}'
            )
        ]
    )
    goal = GoalController(provider)
    goal.set("Deploy to the required service")

    decision = await goal.verify([UserMessage(content="deployment failed")])

    assert decision == GoalDecision(
        status=GoalDecisionStatus.IMPOSSIBLE,
        reason="required service is unavailable",
    )
    assert goal.condition is None
    assert goal.consume_terminal_notice() == (
        "Goal judged impossible: required service is unavailable"
    )


@pytest.mark.asyncio
async def test_invalid_judge_response_fails_open_without_clearing_goal() -> None:
    provider = _Provider(["not json"])
    goal = GoalController(provider)
    goal.set("All tests pass")

    decision = await goal.verify([UserMessage(content="done")])

    assert decision == GoalDecision(
        status=GoalDecisionStatus.UNAVAILABLE,
        reason="judge returned no JSON verdict",
    )
    assert goal.condition == "All tests pass"
    assert goal.consume_terminal_notice() == (
        "Goal verification unavailable: judge returned no JSON verdict"
    )


@pytest.mark.asyncio
async def test_provider_error_fails_open_without_clearing_goal() -> None:
    provider = _Provider([FinalMessage(content="judge timed out", stop_reason="error")])
    goal = GoalController(provider)
    goal.set("All tests pass")

    decision = await goal.verify([UserMessage(content="done")])

    assert decision == GoalDecision(
        status=GoalDecisionStatus.UNAVAILABLE,
        reason="judge timed out",
    )
    assert goal.condition == "All tests pass"
    assert goal.consume_terminal_notice() == (
        "Goal verification unavailable: judge timed out"
    )
