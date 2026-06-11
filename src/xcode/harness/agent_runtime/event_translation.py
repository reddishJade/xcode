"""AgentEvent → StructuredAgentEvent 事件翻译。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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
from ...agent.messages import AgentMessage, AssistantMessage
from ...agent.protocols import AgentToolResult
from xcode.ai.events import ToolCall
from xcode.agent.types import (
    TextContent,
    ToolArguments,
    ToolCallContent,
)

if TYPE_CHECKING:
    from .result import StructuredAgentResult


@dataclass(frozen=True)
class TurnToolResultData:
    tool_call_id: str
    content: str


@dataclass(frozen=True)
class TurnEndData:
    tool_results: list[TurnToolResultData]


@dataclass(frozen=True)
class TextDeltaStructuredEvent:
    type: Literal["text_delta"]
    step: int
    data: str


@dataclass(frozen=True)
class ReasoningDeltaStructuredEvent:
    type: Literal["reasoning_delta"]
    step: int
    data: str


@dataclass(frozen=True)
class MessageStartStructuredEvent:
    type: Literal["message_start"]
    step: int
    data: AgentMessage | None


@dataclass(frozen=True)
class TurnEndStructuredEvent:
    type: Literal["turn_end"]
    step: int
    data: TurnEndData


@dataclass(frozen=True)
class AssistantTextBlock:
    text: str


@dataclass(frozen=True)
class AssistantToolUseBlock:
    id: str
    name: str
    input: ToolArguments


type AssistantEventBlock = AssistantTextBlock | AssistantToolUseBlock


@dataclass(frozen=True)
class AssistantStructuredEvent:
    type: Literal["assistant"]
    step: int
    data: tuple[AssistantEventBlock, ...]


@dataclass(frozen=True)
class ToolUseStructuredEvent:
    type: Literal["tool_use"]
    step: int
    data: ToolCall


@dataclass(frozen=True)
class ToolUpdateData:
    tool_call_id: str
    tool_name: str
    partial_result: str


@dataclass(frozen=True)
class ToolUpdateStructuredEvent:
    type: Literal["tool_update"]
    step: int
    data: ToolUpdateData


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    status: Literal["ok", "error"] = "ok"
    type: str = "tool_result"


@dataclass(frozen=True)
class ToolResultStructuredEvent:
    type: Literal["tool_result"]
    step: int
    data: ToolResultBlock


@dataclass(frozen=True)
class CompactionData:
    messages_removed: int
    messages_after: int
    summary_token_estimate: int
    trigger: str


@dataclass(frozen=True)
class CompactionStructuredEvent:
    type: Literal["compaction"]
    step: int
    data: CompactionData


@dataclass(frozen=True)
class FinalStructuredEvent:
    type: Literal["final"]
    step: int
    data: StructuredAgentResult


type StructuredAgentEvent = (
    MessageStartStructuredEvent
    | TurnEndStructuredEvent
    | TextDeltaStructuredEvent
    | ReasoningDeltaStructuredEvent
    | AssistantStructuredEvent
    | ToolUseStructuredEvent
    | ToolUpdateStructuredEvent
    | ToolResultStructuredEvent
    | CompactionStructuredEvent
    | FinalStructuredEvent
)


@dataclass
class _StreamTranslationState:
    step: int = 0
    text_seen: dict[int, str] = field(default_factory=dict)


def _translate_event(
    event: AgentEvent,
    state: _StreamTranslationState,
) -> StructuredAgentEvent | None:
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
                    return TextDeltaStructuredEvent("text_delta", step, delta)
        return None

    if isinstance(event, MessageStartEvent):
        return MessageStartStructuredEvent("message_start", state.step, event.message)

    if isinstance(event, TurnEndEvent):
        return TurnEndStructuredEvent(
            "turn_end",
            state.step,
            TurnEndData(
                tool_results=[
                    TurnToolResultData(r.tool_call_id, str(r.content))
                    for r in event.tool_results
                ]
            ),
        )

    if isinstance(event, ThinkingUpdateEvent):
        return ReasoningDeltaStructuredEvent(
            "reasoning_delta", state.step, event.reasoning_content
        )

    if isinstance(event, MessageEndEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            blocks = _assistant_event_blocks(msg)
            if blocks:
                return AssistantStructuredEvent("assistant", state.step, blocks)
        return None

    if isinstance(event, ToolExecutionStartEvent):
        tool_use = ToolCall(
            id=event.tool_call_id,
            name=event.tool_name,
            input=event.args or {},
        )
        return ToolUseStructuredEvent("tool_use", state.step, tool_use)

    if isinstance(event, ToolExecutionUpdateEvent):
        return ToolUpdateStructuredEvent(
            "tool_update",
            state.step,
            ToolUpdateData(
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                partial_result=_tool_update_text(event.partial_result),
            ),
        )

    if isinstance(event, ToolExecutionEndEvent):
        return ToolResultStructuredEvent(
            "tool_result",
            state.step,
            ToolResultBlock(
                tool_use_id=event.tool_call_id,
                content=str(event.result.content) if event.result else "",
                status="error" if event.is_error else "ok",
            ),
        )

    if isinstance(event, CompactionEvent):
        return CompactionStructuredEvent(
            "compaction",
            state.step,
            CompactionData(
                messages_removed=event.messages_removed,
                messages_after=event.messages_after,
                summary_token_estimate=event.summary_token_estimate,
                trigger=event.trigger,
            ),
        )

    return None


def _tool_update_text(partial_result: AgentToolResult | None) -> str:
    if partial_result is None:
        return ""
    parts: list[str] = []
    for block in partial_result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCallContent):
            pass
        else:
            parts.append(str(block))
    return "".join(part for part in parts if part)


def _assistant_event_blocks(msg: AssistantMessage) -> tuple[AssistantEventBlock, ...]:
    blocks: list[AssistantEventBlock] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            blocks.append(AssistantTextBlock(block.text))
        elif isinstance(block, ToolCallContent):
            blocks.append(
                AssistantToolUseBlock(block.id, block.name, block.arguments or {})
            )
    return tuple(blocks)
