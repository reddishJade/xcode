"""StructuredAgent 结果类型与转换。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...agent.config import AfterToolCallContext
from ...agent.results import AgentLoopResult, TerminationReason
from ...agent.messages import AssistantMessage
from xcode.ai.events import ToolCall
from xcode.agent.types import TextContent, ToolCallContent
from ..config import ExecutionMode
from .agent_helpers import text_from_blocks, to_dict
from .events import FinalStructuredEvent
from .execution_modes import parse_execution_mode


@dataclass(frozen=True)
class RunState:
    """可序列化的运行状态快照。"""

    messages: list[dict[str, Any]]
    current_mode: ExecutionMode = "act"
    last_agent: str = "main"
    needs_follow_up: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""
        return {
            "messages": self.messages,
            "current_mode": self.current_mode,
            "last_agent": self.last_agent,
            "needs_follow_up": self.needs_follow_up,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "RunState":
        """从 JSON 字典恢复运行状态。"""
        if not isinstance(payload, Mapping):
            return cls(messages=[])
        raw_messages = payload.get("messages", [])
        return cls(
            messages=_message_dicts(raw_messages),
            current_mode=parse_execution_mode(payload.get("current_mode")) or "act",
            last_agent=str(payload.get("last_agent", "main")),
            needs_follow_up=bool(payload.get("needs_follow_up", False)),
        )


@dataclass(frozen=True)
class StructuredAgentResult:
    answer: str
    messages: list[dict[str, Any]]
    steps: int
    tool_calls: list[ToolCall]
    termination_reason: TerminationReason = TerminationReason.COMPLETED
    metrics: dict[str, Any] | None = None
    watchdog_reason: str | None = None
    error_detail: str | None = None
    needs_follow_up: bool = False
    last_agent: str = "main"
    run_state: RunState | None = None

    @property
    def stopped_by_limit(self) -> bool:
        """兼容旧调用方；停止状态以 termination_reason 为准。"""
        return self.termination_reason is TerminationReason.STEP_LIMIT

    @property
    def stopped_by_watchdog(self) -> bool:
        """兼容旧调用方；停止状态以 termination_reason 为准。"""
        return self.termination_reason is TerminationReason.WATCHDOG

    @property
    def stopped_by_error(self) -> bool:
        """兼容旧调用方；停止状态以 termination_reason 为准。"""
        return self.termination_reason is TerminationReason.PROVIDER_ERROR


def _build_structured_result(
    result: AgentLoopResult, max_steps: int, current_mode: ExecutionMode = "act"
) -> StructuredAgentResult:
    """将 AgentLoopResult 转换为 StructuredAgentResult。"""
    answer_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    messages: list[dict[str, Any]] = []
    for msg in result.messages:
        messages.append(to_dict(msg))
        if not isinstance(msg, AssistantMessage):
            continue
        extracted = text_from_blocks(
            [
                {"type": "text", "text": b.text} if isinstance(b, TextContent) else {}
                for b in msg.content
            ]
        )
        if extracted:
            answer_parts.append(extracted)
        for block in msg.content:
            if isinstance(block, ToolCallContent):
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.arguments or {},
                    )
                )

    answer = " ".join(answer_parts)
    metrics = None
    if result.metrics:
        metrics = {
            "llm_calls": result.metrics.llm_calls,
            "tool_calls": result.metrics.tool_calls,
            "estimated_prompt_tokens": result.metrics.input_tokens,
            "estimated_completion_tokens": result.metrics.output_tokens,
            "model_latencies_ms": result.metrics.model_latencies_ms,
            "tool_latencies_ms": result.metrics.tool_latencies_ms,
            "steps": result.metrics.steps,
        }

    if (
        result.termination_reason is TerminationReason.WATCHDOG
        and result.watchdog_reason
    ):
        if answer:
            answer = answer + " " + result.watchdog_reason
        else:
            answer = result.watchdog_reason
    elif result.termination_reason is TerminationReason.STEP_LIMIT and not answer:
        answer = "step limit reached"

    return StructuredAgentResult(
        answer=answer,
        messages=messages,
        steps=result.steps,
        tool_calls=tool_calls,
        last_agent="main",
        termination_reason=result.termination_reason,
        metrics=metrics,
        watchdog_reason=result.watchdog_reason,
        error_detail=result.error_detail,
        run_state=RunState(messages=messages, current_mode=current_mode),
    )


def _message_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _tool_result_text(ctx: AfterToolCallContext) -> str:
    if not ctx.result or not ctx.result.content:
        return ""
    return "".join(c.text for c in ctx.result.content if isinstance(c, TextContent))


def _final_event(step: int, result: StructuredAgentResult) -> FinalStructuredEvent:
    return FinalStructuredEvent("final", step, result)
