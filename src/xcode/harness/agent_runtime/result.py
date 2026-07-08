"""CodingAgentHarness 结果类型与转换。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...agent.results import AgentLoopResult, TerminationReason
from ...agent.messages import AssistantMessage
from xcode.ai.events import ToolCall
from xcode.agent.types import TextContent, ToolCallContent
from xcode.coding_agent.execution_modes import ExecutionMode
from .agent_helpers import text_from_blocks, to_dict
from .events import FinalStructuredEvent
from ..observability import EventCorrelation


def _parse_execution_mode(value: object) -> ExecutionMode | None:
    if not isinstance(value, str):
        return None
    match value:
        case "plan" | "build" | "act":
            return value
        case _:
            return None


@dataclass(frozen=True)
class RunState:
    """可序列化的运行状态快照。"""

    messages: list[dict[str, Any]]
    current_mode: ExecutionMode = "act"
    last_agent: str = "main"
    needs_follow_up: bool = False
    todos: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""
        return {
            "messages": self.messages,
            "current_mode": self.current_mode,
            "last_agent": self.last_agent,
            "needs_follow_up": self.needs_follow_up,
            "todos": self.todos or [],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "RunState":
        """从 JSON 字典恢复运行状态。"""
        if not isinstance(payload, Mapping):
            return cls(messages=[])
        raw_messages = payload.get("messages", [])
        return cls(
            messages=_message_dicts(raw_messages),
            current_mode=_parse_execution_mode(payload.get("current_mode")) or "act",
            last_agent=str(payload.get("last_agent", "main")),
            needs_follow_up=bool(payload.get("needs_follow_up", False)),
            todos=_todo_dicts(payload.get("todos", [])),
        )


@dataclass(frozen=True)
class CodingAgentHarnessResult:
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


def _build_structured_result(
    result: AgentLoopResult,
    max_steps: int,
    current_mode: ExecutionMode = "act",
    todos: list[dict[str, Any]] | None = None,
) -> CodingAgentHarnessResult:
    """将 AgentLoopResult 转换为 CodingAgentHarnessResult。"""
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
            "model_time_ms": sum(result.metrics.model_latencies_ms),
            "tool_time_ms": sum(result.metrics.tool_latencies_ms),
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

    return CodingAgentHarnessResult(
        answer=answer,
        messages=messages,
        steps=result.steps,
        tool_calls=tool_calls,
        last_agent="main",
        termination_reason=result.termination_reason,
        metrics=metrics,
        watchdog_reason=result.watchdog_reason,
        error_detail=result.error_detail,
        run_state=RunState(
            messages=messages,
            current_mode=current_mode,
            todos=todos or [],
        ),
    )


def _message_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _todo_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _final_event(
    step: int,
    result: CodingAgentHarnessResult,
    correlation: EventCorrelation | None = None,
) -> FinalStructuredEvent:
    return FinalStructuredEvent(
        "final",
        step,
        result,
        correlation or EventCorrelation(),
    )
