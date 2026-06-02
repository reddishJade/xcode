from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallEvent,
)

from .types import ContentBlock, TextContent, ToolCallBlock


@dataclass(frozen=True)
class ProviderOutputDelta:
    kind: Literal["text", "reasoning"]
    chunk: str


@dataclass(frozen=True)
class ProviderResponse:
    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str | None = None
    reasoning_content: str | None = None
    deltas: list[ProviderOutputDelta] = field(default_factory=list)


def provider_events_to_response(events: Iterable[ProviderEvent]) -> ProviderResponse:
    """将 provider stream 事件聚合为 Agent assistant response。"""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    content: list[ContentBlock] = []
    deltas: list[ProviderOutputDelta] = []
    stop_reason: str | None = None

    for event in events:
        if isinstance(event, ReasoningDelta):
            if event.chunk:
                reasoning_parts.append(event.chunk)
                deltas.append(ProviderOutputDelta("reasoning", event.chunk))
        elif isinstance(event, TextDelta):
            if event.chunk:
                text_parts.append(event.chunk)
                deltas.append(ProviderOutputDelta("text", event.chunk))
        elif isinstance(event, ToolCallEvent):
            content.extend(
                ToolCallBlock(
                    id=call.id,
                    name=call.name,
                    arguments=dict(call.input),
                )
                for call in event.calls
            )
        elif isinstance(event, FinalMessage):
            if event.content and not text_parts:
                text_parts.append(event.content)
            if event.stop_reason:
                stop_reason = event.stop_reason

    if text_parts:
        content.insert(0, TextContent(text="".join(text_parts)))

    return ProviderResponse(
        content=content,
        stop_reason=stop_reason,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
        deltas=deltas,
    )
