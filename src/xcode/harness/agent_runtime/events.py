"""AgentEvent → StructuredAgentEvent 事件翻译。"""

from __future__ import annotations

import json
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
from ..session_todo import TodoItem
from ..observability import EventCorrelation, RuntimeCorrelation

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
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class ReasoningDeltaStructuredEvent:
    type: Literal["reasoning_delta"]
    step: int
    data: str
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class MessageStartStructuredEvent:
    type: Literal["message_start"]
    step: int
    data: AgentMessage | None
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class TurnEndStructuredEvent:
    type: Literal["turn_end"]
    step: int
    data: TurnEndData
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


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
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class ToolUseStructuredEvent:
    type: Literal["tool_use"]
    step: int
    data: ToolCall
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


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
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


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
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class TodoUpdateStructuredEvent:
    """会话待办完整替换事件。"""

    type: Literal["todo_update"]
    step: int
    data: tuple[TodoItem, ...]
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


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
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class FinalStructuredEvent:
    type: Literal["final"]
    step: int
    data: StructuredAgentResult
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


type StructuredAgentEvent = (
    MessageStartStructuredEvent
    | TurnEndStructuredEvent
    | TextDeltaStructuredEvent
    | ReasoningDeltaStructuredEvent
    | AssistantStructuredEvent
    | ToolUseStructuredEvent
    | ToolUpdateStructuredEvent
    | ToolResultStructuredEvent
    | TodoUpdateStructuredEvent
    | CompactionStructuredEvent
    | FinalStructuredEvent
)


@dataclass
class _StreamTranslationState:
    step: int = 0
    text_seen: dict[int, str] = field(default_factory=dict)
    correlation: RuntimeCorrelation = field(
        default_factory=lambda: RuntimeCorrelation("local")
    )


def _translate_event(
    event: AgentEvent,
    state: _StreamTranslationState,
) -> StructuredAgentEvent | list[StructuredAgentEvent] | None:
    if isinstance(event, AgentStartEvent):
        return None

    if isinstance(event, TurnStartEvent):
        state.step += 1
        return None

    if isinstance(event, MessageUpdateEvent):
        return _translate_message_update(event, state)

    if isinstance(event, MessageStartEvent):
        return _translate_message_start(event, state)

    if isinstance(event, TurnEndEvent):
        return _translate_turn_end(event, state)

    if isinstance(event, ThinkingUpdateEvent):
        return _translate_thinking_update(event, state)

    if isinstance(event, MessageEndEvent):
        return _translate_message_end(event, state)

    if isinstance(event, ToolExecutionStartEvent):
        return _translate_tool_execution_start(event, state)

    if isinstance(event, ToolExecutionUpdateEvent):
        return _translate_tool_execution_update(event, state)

    if isinstance(event, ToolExecutionEndEvent):
        return _translate_tool_execution_end(event, state)

    if isinstance(event, CompactionEvent):
        return _translate_compaction(event, state)

    return None


def _translate_message_update(
    event: MessageUpdateEvent,
    state: _StreamTranslationState,
) -> StructuredAgentEvent | None:
    msg = event.message
    if not isinstance(msg, AssistantMessage) or not msg.content:
        return None
    for block in msg.content:
        if isinstance(block, TextContent) and block.text:
            step = state.step
            prev = state.text_seen.get(step, "")
            full = block.text
            delta = full[len(prev) :]
            if not delta:
                return None
            state.text_seen[step] = full
            return TextDeltaStructuredEvent(
                "text_delta",
                step,
                delta,
                state.correlation.snapshot(),
            )
    return None


def _translate_message_start(
    event: MessageStartEvent,
    state: _StreamTranslationState,
) -> MessageStartStructuredEvent:
    return MessageStartStructuredEvent(
        "message_start",
        state.step,
        event.message,
        state.correlation.snapshot(),
    )


def _translate_turn_end(
    event: TurnEndEvent,
    state: _StreamTranslationState,
) -> TurnEndStructuredEvent:
    return TurnEndStructuredEvent(
        "turn_end",
        state.step,
        TurnEndData(
            tool_results=[
                TurnToolResultData(r.tool_call_id, str(r.content))
                for r in event.tool_results
            ]
        ),
        state.correlation.snapshot(),
    )


def _translate_thinking_update(
    event: ThinkingUpdateEvent,
    state: _StreamTranslationState,
) -> ReasoningDeltaStructuredEvent:
    return ReasoningDeltaStructuredEvent(
        "reasoning_delta",
        state.step,
        event.reasoning_content,
        state.correlation.snapshot(),
    )


def _translate_message_end(
    event: MessageEndEvent,
    state: _StreamTranslationState,
) -> AssistantStructuredEvent | None:
    msg = event.message
    if not isinstance(msg, AssistantMessage) or not msg.content:
        return None
    blocks = _assistant_event_blocks(msg)
    if not blocks:
        return None
    return AssistantStructuredEvent(
        "assistant",
        state.step,
        blocks,
        state.correlation.snapshot(),
    )


def _translate_tool_execution_start(
    event: ToolExecutionStartEvent,
    state: _StreamTranslationState,
) -> ToolUseStructuredEvent:
    tool_use = ToolCall(
        id=event.tool_call_id,
        name=event.tool_name,
        input=event.args or {},
    )
    return ToolUseStructuredEvent(
        "tool_use",
        state.step,
        tool_use,
        state.correlation.snapshot(event.tool_call_id),
    )


def _translate_tool_execution_update(
    event: ToolExecutionUpdateEvent,
    state: _StreamTranslationState,
) -> ToolUpdateStructuredEvent:
    return ToolUpdateStructuredEvent(
        "tool_update",
        state.step,
        ToolUpdateData(
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            partial_result=_tool_update_text(event.partial_result),
        ),
        state.correlation.snapshot(event.tool_call_id),
    )


def _translate_tool_execution_end(
    event: ToolExecutionEndEvent,
    state: _StreamTranslationState,
) -> StructuredAgentEvent | list[StructuredAgentEvent]:
    result_event = ToolResultStructuredEvent(
        "tool_result",
        state.step,
        ToolResultBlock(
            tool_use_id=event.tool_call_id,
            content=str(event.result.content) if event.result else "",
            status="error" if event.is_error else "ok",
        ),
        state.correlation.snapshot(event.tool_call_id),
    )
    todo_items = _todo_items_from_result(event)
    if todo_items is None:
        return result_event
    return [
        result_event,
        TodoUpdateStructuredEvent(
            "todo_update",
            state.step,
            todo_items,
            state.correlation.snapshot(event.tool_call_id),
        ),
    ]


def _translate_compaction(
    event: CompactionEvent,
    state: _StreamTranslationState,
) -> CompactionStructuredEvent:
    return CompactionStructuredEvent(
        "compaction",
        state.step,
        CompactionData(
            messages_removed=event.messages_removed,
            messages_after=event.messages_after,
            summary_token_estimate=event.summary_token_estimate,
            trigger=event.trigger,
        ),
        state.correlation.snapshot(),
    )


def _todo_items_from_result(
    event: ToolExecutionEndEvent,
) -> tuple[TodoItem, ...] | None:
    """从成功的 update_todo 工具结果构建结构化事件。"""
    if event.tool_name != "update_todo" or event.is_error or event.result is None:
        return None
    try:
        payload = json.loads(str(event.result.content))
    except json.JSONDecodeError:
        return None
    raw_items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        return None
    items: list[TodoItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            return None
        item_id = raw_item.get("id")
        content = raw_item.get("content")
        status = raw_item.get("status")
        if (
            not isinstance(item_id, str)
            or not isinstance(content, str)
            or status not in {"pending", "in_progress", "completed"}
        ):
            return None
        items.append(TodoItem(item_id, content, status))
    return tuple(items)


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
