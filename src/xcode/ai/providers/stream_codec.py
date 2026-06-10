"""流式事件解码。

处理 Chat Completions 流式响应，
将原始 chunk 转换为统一的 ProviderEvent。
"""

from __future__ import annotations

import orjson
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Protocol

from xcode.ai.events import (
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    UsageUpdate,
)


# ── OpenAI 对象 Protocol ──


class _Usage(Protocol):
    @property
    def prompt_tokens(self) -> int: ...
    @property
    def completion_tokens(self) -> int: ...


class _ChoiceDeltaToolCallFunction(Protocol):
    @property
    def name(self) -> str | None: ...
    @property
    def arguments(self) -> str | None: ...


class _ChoiceDeltaToolCall(Protocol):
    @property
    def index(self) -> int: ...
    @property
    def id(self) -> str | None: ...
    @property
    def function(self) -> _ChoiceDeltaToolCallFunction | None: ...


class _ChoiceDelta(Protocol):
    @property
    def content(self) -> str | None: ...
    @property
    def tool_calls(self) -> Sequence[_ChoiceDeltaToolCall] | None: ...


class _Choice(Protocol):
    @property
    def delta(self) -> _ChoiceDelta: ...


class _ChatCompletionChunk(Protocol):
    @property
    def usage(self) -> _Usage | None: ...
    @property
    def choices(self) -> Sequence[_Choice]: ...


# ── 工具调用解析 ──


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """解析工具调用参数 JSON。"""
    try:
        result = orjson.loads((raw_arguments or "{}").encode())
        return result if isinstance(result, dict) else {"input": str(result)}
    except orjson.JSONDecodeError:
        return {"input": raw_arguments}


# ── 流式事件解码 ──


def chat_stream_to_events(
    stream: Iterable[_ChatCompletionChunk],
) -> Iterator[TextDelta | ReasoningDelta | ToolCallEvent | UsageUpdate]:
    """Yields provider events: TextDelta, ReasoningDelta, ToolCallEvent, UsageUpdate."""
    calls: dict[int, dict[str, str]] = defaultdict(
        lambda: {"id": "", "name": "", "arguments": ""}
    )
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield UsageUpdate(
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
            )

        choices = chunk.choices
        if not choices:
            continue
        delta = choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ReasoningDelta(chunk=str(reasoning))
        text = delta.content
        if text:
            yield TextDelta(chunk=str(text))
        for call in delta.tool_calls or []:
            index = call.index
            current = calls[index]
            if call.id is not None:
                current["id"] = call.id
            func = call.function
            if func is not None:
                if func.name is not None:
                    current["name"] = func.name
                if func.arguments is not None:
                    current["arguments"] += func.arguments

    ready = [
        ToolCall(id=c["id"], name=c["name"], input=parse_tool_arguments(c["arguments"]))
        for _, c in sorted(calls.items())
    ]
    if ready:
        yield ToolCallEvent(calls=ready)
