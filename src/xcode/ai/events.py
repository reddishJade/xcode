from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

"""Provider 流式事件协议。"""

type Message = dict[str, Any]
type StopReason = Literal[
    "end_turn",
    "max_tokens",
    "tool_use",
    "max_steps",
    "cancelled",
    "error",
    "stop_sequence",
    "aborted",
]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class TextDelta:
    chunk: str


@dataclass(frozen=True)
class ToolCallEvent:
    calls: list[ToolCall]


@dataclass(frozen=True)
class UsageUpdate:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FinalMessage:
    content: str
    stop_reason: StopReason


@dataclass(frozen=True)
class ReasoningDelta:
    chunk: str


type ProviderEvent = (
    TextDelta | ToolCallEvent | UsageUpdate | FinalMessage | ReasoningDelta
)
