"""独立 auto-reviewer 的结构化 assessment 测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast

import pytest

from xcode.agent.types import ApprovalRequest, ToolSpec
from xcode.ai.events import FinalMessage, Message, ProviderEvent, TextDelta
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.security.approval_reviewer import (
    AutoApprovalReviewer,
    AutoReviewVerdict,
    parse_auto_review_verdict,
)


class _ReviewerProvider:
    model = "reviewer-model"
    base_url = "https://reviewer.invalid"
    transport = "test"
    thinking = False
    reasoning_effort = None

    def __init__(self, response: str, *, error: bool = False) -> None:
        self.response = response
        self.error = error
        self.requests: list[tuple[list[Message], list[ToolDefinition]]] = []
        self.options: list[StreamOptions | None] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        del kwargs
        self.requests.append((deepcopy(messages), deepcopy(tools)))
        self.options.append(options)
        if self.error:
            yield FinalMessage(content=self.response, stop_reason="error")
            return
        yield TextDelta(self.response)
        yield FinalMessage(content="", stop_reason="end_turn")


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        tool=ToolSpec(
            name="bash",
            description="Run a shell command",
            input_hint="",
            handler=lambda _data, _update: "",
        ),
        action_input={"command": "pytest -q"},
        allowed_scopes=("once", "session", "permanent"),
        reason="shell command requires approval",
        transcript=("<user trust=trusted>\nRun the project's focused tests.\n</user>"),
        working_directory="/workspace/project",
        turn_id="session:turn:1",
    )


def test_parse_low_risk_allow_uses_guardian_defaults() -> None:
    verdict = parse_auto_review_verdict('{"outcome":"allow"}')

    assert verdict == AutoReviewVerdict(
        "allow",
        "low",
        "unknown",
        "Auto-review returned a low-risk allow decision.",
    )


def test_parse_medium_risk_allow_is_not_rejected_by_local_threshold() -> None:
    verdict = parse_auto_review_verdict(
        "```json\n"
        '{"outcome":"allow","risk_level":"medium",'
        '"user_authorization":"medium","rationale":"bounded side effect"}\n```'
    )

    assert verdict == AutoReviewVerdict(
        "allow",
        "medium",
        "medium",
        "bounded side effect",
    )


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"outcome":"maybe"}',
        '{"outcome":"allow","risk_level":"extreme"}',
        '{"outcome":"allow","user_authorization":"explicit"}',
        '{"outcome":"allow","scope":"permanent"}',
        '{"outcome":"allow"}{"outcome":"deny"}',
    ],
)
def test_parse_auto_review_verdict_rejects_invalid_output(response: str) -> None:
    with pytest.raises(ValueError):
        parse_auto_review_verdict(response)


def test_auto_reviewer_allows_once_and_receives_full_evidence() -> None:
    provider = _ReviewerProvider(
        '{"outcome":"allow","risk_level":"low",'
        '"user_authorization":"high","rationale":"Focused local validation."}'
    )
    reviewer = AutoApprovalReviewer(cast(Any, provider))
    try:
        result = reviewer(_request())
    finally:
        reviewer.close()

    assert result.decision == "allow"
    assert result.scope == "once"
    assert result.status == "completed"
    assert result.risk == "low"
    assert result.authorization == "high"
    assert result.rationale == "Focused local validation."
    assert provider.requests[0][1] == []
    assert provider.options == [
        StreamOptions(
            temperature=0,
            max_tokens=512,
            timeout_ms=90_000,
            max_retries=1,
        )
    ]
    prompt = str(provider.requests[0][0][-1]["content"])
    assert "pytest -q" in prompt
    assert "Run the project's focused tests." in prompt
    assert "/workspace/project" in prompt
    assert "session:turn:1" in prompt


def test_auto_reviewer_retries_then_fails_closed_on_provider_error() -> None:
    provider = _ReviewerProvider("review service failed", error=True)
    reviewer = AutoApprovalReviewer(cast(Any, provider))
    try:
        result = reviewer(_request())
    finally:
        reviewer.close()

    assert result.decision == "deny"
    assert result.scope == "once"
    assert result.status == "failed"
    assert "failed" in result.suggestion
    assert len(provider.requests) == 3
