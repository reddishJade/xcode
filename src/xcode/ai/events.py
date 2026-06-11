"""Provider 流式事件协议。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    """单次工具调用。"""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class TextDelta:
    """文本增量。"""

    chunk: str


@dataclass(frozen=True)
class ToolCallEvent:
    """一组工具调用。"""

    calls: list[ToolCall]


@dataclass(frozen=True)
class UsageUpdate:
    """token 用量更新。"""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FinalMessage:
    """流式响应终止消息。"""

    content: str
    stop_reason: StopReason


@dataclass(frozen=True)
class ReasoningDelta:
    """推理内容增量。"""

    chunk: str


type ProviderEvent = (
    TextDelta | ToolCallEvent | UsageUpdate | FinalMessage | ReasoningDelta
)
