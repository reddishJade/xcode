"""StructuredAgent 结果类型与转换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...agent.config import AfterToolCallContext, AgentLoopResult
from ...agent.messages import AssistantMessage
from xcode.ai.events import ToolCall as ToolUseBlock
from xcode.ai.types import TextContent, ToolCallContent
from .agent_helpers import text_from_blocks, to_dict
from .event_translation import StructuredAgentEvent


@dataclass(frozen=True)
class StructuredAgentResult:
    answer: str
    messages: list[dict[str, Any]]
    steps: int
    tool_calls: list[ToolUseBlock]
    stopped_by_limit: bool = False
    metrics: dict[str, Any] | None = None
    stopped_by_watchdog: bool = False
    watchdog_reason: str | None = None
    needs_follow_up: bool = False


def _build_structured_result(
    result: AgentLoopResult, max_steps: int
) -> StructuredAgentResult:
    """将 AgentLoopResult 转换为 StructuredAgentResult。"""
    answer_parts: list[str] = []
    tool_calls: list[ToolUseBlock] = []
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
                    ToolUseBlock(
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

    if result.stopped_by_watchdog and result.watchdog_reason:
        if answer:
            answer = answer + " " + result.watchdog_reason
        else:
            answer = result.watchdog_reason
    elif result.stopped_by_limit and not answer:
        answer = "step limit reached"

    return StructuredAgentResult(
        answer=answer,
        messages=messages,
        steps=result.steps,
        tool_calls=tool_calls,
        stopped_by_limit=result.stopped_by_limit,
        metrics=metrics,
        stopped_by_watchdog=result.stopped_by_watchdog,
        watchdog_reason=result.watchdog_reason,
    )


def _tool_result_text(ctx: AfterToolCallContext) -> str:
    if not ctx.result or not ctx.result.content:
        return ""
    return "".join(c.text for c in ctx.result.content if isinstance(c, TextContent))


def _final_event(step: int, result: StructuredAgentResult) -> StructuredAgentEvent:
    return StructuredAgentEvent("final", step, result)
