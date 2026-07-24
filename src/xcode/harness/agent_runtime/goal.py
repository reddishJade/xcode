"""长任务停止条件的独立完成度验证。"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import json

from xcode.agent._codec import convert_to_llm
from xcode.agent.messages import AgentMessage
from xcode.ai.events import FinalMessage, TextDelta
from xcode.ai.providers.base import ModelProvider
from xcode.ai.types import StreamOptions

_JUDGE_SYSTEM = """\
You independently verify a coding agent's stop condition.
Use only evidence in the supplied transcript, including actual tool results.
Return exactly one JSON object:
{"ok": true, "reason": "specific evidence"}
{"ok": false, "reason": "specific missing evidence"}
{"ok": false, "impossible": true, "reason": "why it cannot be completed"}
Do not trust the working agent's claim without transcript evidence.
Use impossible only when the condition genuinely cannot be completed."""


@dataclass(frozen=True)
class GoalVerdict:
    """独立 judge 对停止条件的判定。"""

    ok: bool
    reason: str
    impossible: bool = False


class GoalDecisionStatus(StrEnum):
    """一次 Goal 验收的明确结果。"""

    NO_GOAL = "no_goal"
    PAUSED = "paused"
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    IMPOSSIBLE = "impossible"
    UNAVAILABLE = "unavailable"
    REENTRY_LIMIT = "reentry_limit"


@dataclass(frozen=True)
class GoalDecision:
    """GoalController 对本次停止请求作出的决定。"""

    status: GoalDecisionStatus
    reason: str | None = None
    feedback: str | None = None


@dataclass(frozen=True)
class GoalState:
    """可随 session 保存和恢复的 Goal 状态。"""

    condition: str | None = None
    paused: bool = False
    reacts: int = 0

    def to_dict(self) -> dict[str, str | bool | int | None]:
        return {
            "condition": self.condition,
            "paused": self.paused,
            "reacts": self.reacts,
        }

    @classmethod
    def from_dict(cls, payload: object) -> GoalState:
        if not isinstance(payload, Mapping):
            return cls()
        raw_condition = payload.get("condition")
        condition = (
            raw_condition.strip()
            if isinstance(raw_condition, str) and raw_condition.strip()
            else None
        )
        raw_reacts = payload.get("reacts", 0)
        reacts = (
            raw_reacts
            if isinstance(raw_reacts, int) and not isinstance(raw_reacts, bool)
            else 0
        )
        return cls(
            condition=condition,
            paused=condition is not None and bool(payload.get("paused", False)),
            reacts=max(0, reacts),
        )


class GoalController:
    """保存当前停止条件，并在主 Agent 尝试结束时独立验收。"""

    def __init__(
        self,
        provider: ModelProvider | Callable[[], ModelProvider],
        *,
        max_reacts: int = 3,
    ) -> None:
        self._provider = provider
        self._max_reacts = max(1, max_reacts)
        self._condition: str | None = None
        self._paused = False
        self._reacts = 0
        self._terminal_notice: str | None = None

    @property
    def condition(self) -> str | None:
        return self._condition

    @property
    def paused(self) -> bool:
        return self._condition is not None and self._paused

    @property
    def state(self) -> GoalState:
        return GoalState(
            condition=self._condition,
            paused=self.paused,
            reacts=self._reacts,
        )

    def set(self, condition: str) -> None:
        normalized = condition.strip()
        if not normalized:
            raise ValueError("goal condition must not be empty")
        self._condition = normalized
        self._paused = False
        self._reacts = 0
        self._terminal_notice = None

    def pause(self) -> None:
        if self._condition is None:
            raise ValueError("cannot pause without an active goal")
        self._paused = True

    def resume(self) -> None:
        if self._condition is None:
            raise ValueError("cannot resume without an active goal")
        self._paused = False

    def clear(self) -> None:
        self._condition = None
        self._paused = False
        self._reacts = 0

    def restore(self, state: GoalState) -> None:
        self._condition = state.condition
        self._paused = state.condition is not None and state.paused
        self._reacts = state.reacts if state.condition is not None else 0
        self._terminal_notice = None

    def consume_terminal_notice(self) -> str | None:
        notice = self._terminal_notice
        self._terminal_notice = None
        return notice

    async def verify(self, messages: list[AgentMessage]) -> GoalDecision:
        """验收停止条件，并返回无歧义的结构化决定。"""
        condition = self._condition
        if condition is None:
            return GoalDecision(GoalDecisionStatus.NO_GOAL)
        if self._paused:
            return GoalDecision(GoalDecisionStatus.PAUSED)
        try:
            provider = self._provider() if callable(self._provider) else self._provider
            verdict = await _evaluate(provider, condition, messages)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            reason = str(exc)
            self._terminal_notice = f"Goal verification unavailable: {reason}"
            return GoalDecision(GoalDecisionStatus.UNAVAILABLE, reason=reason)
        if self._condition != condition:
            return GoalDecision(GoalDecisionStatus.NO_GOAL)
        if self._paused:
            return GoalDecision(GoalDecisionStatus.PAUSED)
        if verdict.ok:
            self.clear()
            return GoalDecision(
                GoalDecisionStatus.SATISFIED,
                reason=verdict.reason,
            )
        if verdict.impossible:
            self._terminal_notice = f"Goal judged impossible: {verdict.reason}"
            self.clear()
            return GoalDecision(
                GoalDecisionStatus.IMPOSSIBLE,
                reason=verdict.reason,
            )
        self._reacts += 1
        if self._reacts > self._max_reacts:
            self._terminal_notice = (
                f"Goal remains unsatisfied after {self._max_reacts} re-entries: "
                f"{verdict.reason}"
            )
            self.clear()
            return GoalDecision(
                GoalDecisionStatus.REENTRY_LIMIT,
                reason=verdict.reason,
            )
        feedback = (
            "<goal-verification>\n"
            f"Stop condition is not yet satisfied: {verdict.reason}\n"
            "Continue working and produce transcript evidence before stopping.\n"
            "</goal-verification>"
        )
        return GoalDecision(
            GoalDecisionStatus.UNSATISFIED,
            reason=verdict.reason,
            feedback=feedback,
        )

    async def completion_feedback(
        self,
        messages: list[AgentMessage],
    ) -> str | None:
        """将结构化 Goal 决定适配为通用 Agent Loop 的反馈接口。"""
        return (await self.verify(messages)).feedback


async def _evaluate(
    provider: ModelProvider,
    condition: str,
    messages: list[AgentMessage],
) -> GoalVerdict:
    request = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        *convert_to_llm(messages),
        {
            "role": "user",
            "content": (
                f"Has this stop condition been satisfied?\n\nCondition: {condition}"
            ),
        },
    ]
    text: list[str] = []
    error: str | None = None
    async for event in provider.stream(
        messages=request,
        tools=[],
        options=StreamOptions(temperature=0, max_tokens=512),
    ):
        if isinstance(event, TextDelta):
            text.append(event.chunk)
        elif isinstance(event, FinalMessage):
            if event.stop_reason == "error":
                error = event.content or "provider error"
            elif event.content and not text:
                text.append(event.content)
    if error is not None:
        raise RuntimeError(error)
    return _parse_verdict("".join(text))


def _parse_verdict(text: str) -> GoalVerdict:
    payloads = _decode_json_objects(text)
    if not payloads:
        raise ValueError("judge returned no JSON verdict")
    if len(payloads) > 1:
        raise ValueError("judge returned multiple JSON verdicts")
    payload = payloads[0]
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise ValueError("judge verdict must contain boolean ok")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("judge verdict must contain reason")
    return GoalVerdict(
        ok=payload["ok"],
        impossible=bool(payload.get("impossible", False)),
        reason=reason,
    )


def _decode_json_objects(text: str) -> list[object]:
    """从带少量包装文本的响应中解码顶层 JSON 对象。"""
    decoder = json.JSONDecoder()
    payloads: list[object] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        payloads.append(payload)
        cursor = end
    return payloads
