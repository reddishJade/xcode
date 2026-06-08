"""AgentEvent → StructuredAgentEvent 事件翻译。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...agent.events import (
    AgentEvent,
    AgentStartEvent,
    CompactionEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ThinkingUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from ...agent.messages import AssistantMessage
from xcode.ai.events import ToolCall as ToolUseBlock
from xcode.agent.types import TextContent, ToolCallContent


@dataclass(frozen=True)
class StructuredAgentEvent:
    type: str
    step: int
    data: Any


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    status: str = "ok"
    type: str = "tool_result"


@dataclass
class _StreamTranslationState:
    step: int = 0
    text_seen: dict[int, str] = field(default_factory=dict)


def _translate_event(
    event: AgentEvent,
    state: _StreamTranslationState,
) -> StructuredAgentEvent | list[StructuredAgentEvent] | None:
    """将 AgentEvent 翻译为 StructuredAgentEvent。"""
    if isinstance(event, AgentStartEvent):
        return None

    if isinstance(event, TurnStartEvent):
        state.step += 1
        return None

    if isinstance(event, MessageUpdateEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    step = state.step
                    prev = state.text_seen.get(step, "")
                    full = block.text
                    delta = full[len(prev) :]
                    if not delta:
                        return None
                    state.text_seen[step] = full
                    return StructuredAgentEvent("text_delta", step, delta)
        return None

    if isinstance(event, MessageStartEvent):
        return StructuredAgentEvent("message_start", state.step, event.message)

    if isinstance(event, TurnEndEvent):
        return StructuredAgentEvent(
            "turn_end",
            state.step,
            {
                "tool_results": [
                    {"tool_call_id": r.tool_call_id, "content": str(r.content)}
                    for r in event.tool_results
                ]
            },
        )

    if isinstance(event, ThinkingUpdateEvent):
        return StructuredAgentEvent(
            "reasoning_delta", state.step, event.reasoning_content
        )

    if isinstance(event, MessageEndEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            blocks = _assistant_to_raw_blocks(msg)
            if blocks:
                return StructuredAgentEvent("assistant", state.step, blocks)
        return None

    if isinstance(event, ToolExecutionStartEvent):
        tool_use = ToolUseBlock(
            id=event.tool_call_id,
            name=event.tool_name,
            input=event.args or {},
        )
        return StructuredAgentEvent("tool_use", state.step, tool_use)

    if isinstance(event, ToolExecutionUpdateEvent):
        return StructuredAgentEvent(
            "tool_update",
            state.step,
            {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "partial_result": str(event.partial_result)
                if event.partial_result
                else "",
            },
        )

    if isinstance(event, ToolExecutionEndEvent):
        return StructuredAgentEvent(
            "tool_result",
            state.step,
            ToolResultBlock(
                tool_use_id=event.tool_call_id,
                content=str(event.result.content) if event.result else "",
                status="error" if event.is_error else "ok",
            ),
        )

    if isinstance(event, CompactionEvent):
        return StructuredAgentEvent(
            "compaction",
            state.step,
            {
                "messages_removed": event.messages_removed,
                "messages_after": event.messages_after,
                "summary_token_estimate": event.summary_token_estimate,
                "trigger": event.trigger,
            },
        )

    return None


def _assistant_to_raw_blocks(msg: AssistantMessage) -> list[dict[str, Any]]:
    """将 AssistantMessage 转换为 raw block 列表。"""
    blocks: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallContent):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments or {},
                }
            )
    return blocks
