"""Provider 流式事件协议。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

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


class ToolCall(BaseModel):
    """单次工具调用。"""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    input: dict[str, Any]


class TextDelta(BaseModel):
    """文本增量。"""

    model_config = ConfigDict(frozen=True)

    chunk: str


class ToolCallEvent(BaseModel):
    """一组工具调用。"""

    model_config = ConfigDict(frozen=True)

    calls: list[ToolCall]


class UsageUpdate(BaseModel):
    """token 用量更新。"""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int


class FinalMessage(BaseModel):
    """流式响应终止消息。"""

    model_config = ConfigDict(frozen=True)

    content: str
    stop_reason: StopReason


class ReasoningDelta(BaseModel):
    """推理内容增量。"""

    model_config = ConfigDict(frozen=True)

    chunk: str


type ProviderEvent = (
    TextDelta | ToolCallEvent | UsageUpdate | FinalMessage | ReasoningDelta
)
