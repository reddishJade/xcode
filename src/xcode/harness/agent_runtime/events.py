from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

"""Agent runtime 与 provider 之间的内部事件协议。"""

Message: TypeAlias = dict[str, Any]
StopReason: TypeAlias = Literal[
    "end_turn", "tool_use", "max_steps", "cancelled", "error", "max_tokens"
]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    status: str = "ok"
    elapsed_ms: float | None = None


@dataclass(frozen=True)
class TextDelta:
    chunk: str


@dataclass(frozen=True)
class ToolCallReady:
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


ProviderEvent: TypeAlias = (
    TextDelta | ToolCallReady | UsageUpdate | FinalMessage | ReasoningDelta
)
