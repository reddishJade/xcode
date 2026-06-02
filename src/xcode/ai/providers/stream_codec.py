"""流式事件解码。

处理 Chat Completions 和 Responses API 的流式响应，
将原始 chunk 转换为统一的 ProviderEvent。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from xcode.ai.events import (
    FinalMessage,
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
    def tool_calls(self) -> list[_ChoiceDeltaToolCall] | None: ...


class _Choice(Protocol):
    @property
    def delta(self) -> _ChoiceDelta: ...


class _ChatCompletionChunk(Protocol):
    @property
    def usage(self) -> _Usage | None: ...
    @property
    def choices(self) -> list[_Choice]: ...


class _ResponseContent(Protocol):
    @property
    def text(self) -> str | None: ...


class _ResponseOutputItem(Protocol):
    @property
    def type(self) -> str: ...
    @property
    def content(self) -> list[_ResponseContent] | None: ...
    @property
    def call_id(self) -> str | None: ...
    @property
    def id(self) -> str | None: ...
    @property
    def name(self) -> str | None: ...
    @property
    def arguments(self) -> str | None: ...


class _Response(Protocol):
    @property
    def output_text(self) -> str | None: ...
    @property
    def output(self) -> list[_ResponseOutputItem] | None: ...
    @property
    def id(self) -> str | None: ...


class _ResponseStreamEvent(Protocol):
    @property
    def type(self) -> str: ...
    @property
    def delta(self) -> str | None: ...
    @property
    def response(self) -> _Response | None: ...


# ── 工具调用解析 ──


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        result = json.loads(raw_arguments or "{}")
        return result if isinstance(result, dict) else {"input": str(result)}
    except json.JSONDecodeError:
        return {"input": raw_arguments}


def tool_call_from_response_item(item: _ResponseOutputItem) -> dict[str, Any]:
    return {
        "id": str(item.call_id or item.id or ""),
        "name": str(item.name or ""),
        "input": parse_tool_arguments(str(item.arguments or "{}")),
    }


# ── 流式事件解码 ──


def chat_stream_to_events(
    stream: Iterable[_ChatCompletionChunk],
) -> Iterator[TextDelta | ReasoningDelta | ToolCallEvent | UsageUpdate]:
    """Yields provider events: TextDelta, ReasoningDelta, ToolCallEvent, UsageUpdate."""
    calls: dict[int, dict[str, str]] = defaultdict(
        lambda: {"id": "", "name": "", "arguments": ""}
    )
    for chunk in stream:
        usage = chunk.usage
        if usage is not None:
            yield UsageUpdate(usage.prompt_tokens or 0, usage.completion_tokens or 0)

        choices = chunk.choices
        if not choices:
            continue
        delta = choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ReasoningDelta(str(reasoning))
        text = delta.content
        if text:
            yield TextDelta(str(text))
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
        yield ToolCallEvent(ready)


def responses_stream_to_events(
    stream: Iterable[_ResponseStreamEvent],
) -> Iterator[TextDelta | ReasoningDelta | ToolCallEvent | UsageUpdate | FinalMessage]:
    """处理 Responses API 流式事件。"""
    pending_calls: dict[int, dict[str, str]] = {}
    accumulated_text = ""
    completed = False
    response_text = ""

    for event in stream:
        event_type = event.type

        if event_type == "response.output_text.delta":
            text = event.delta
            if text:
                accumulated_text += str(text)
                yield TextDelta(str(text))
        elif event_type == "response.reasoning_summary_text.delta":
            text = event.delta
            if text:
                yield ReasoningDelta(str(text))
        elif event_type == "response.function_call_arguments.delta":
            index = getattr(event, "output_index", 0)
            delta = getattr(event, "delta", "")
            if index not in pending_calls:
                pending_calls[index] = {"id": "", "name": "", "arguments": ""}
            pending_calls[index]["arguments"] += delta
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if item is not None:
                item_type = getattr(item, "type", "")
                if item_type == "function_call":
                    index = getattr(event, "output_index", 0)
                    if index not in pending_calls:
                        pending_calls[index] = {"id": "", "name": "", "arguments": ""}
                    call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
                    name = getattr(item, "name", "")
                    arguments = getattr(item, "arguments", "")
                    pending_calls[index]["id"] = str(call_id)
                    pending_calls[index]["name"] = str(name)
                    if arguments:
                        pending_calls[index]["arguments"] = str(arguments)
        elif event_type == "response.completed":
            completed = True
            response = getattr(event, "response", None)
            if response is not None:
                response_text = str(getattr(response, "output_text", "") or "")
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
                    yield UsageUpdate(int(input_tokens), int(output_tokens))

    if pending_calls:
        ready = [
            ToolCall(id=c["id"], name=c["name"], input=parse_tool_arguments(c["arguments"]))
            for _, c in sorted(pending_calls.items())
        ]
        yield ToolCallEvent(ready)
    elif completed:
        final_text = response_text or accumulated_text
        yield FinalMessage(final_text, "end_turn")


def responses_to_events(
    response: _Response,
) -> list[TextDelta | FinalMessage | ToolCallEvent]:
    events: list[TextDelta | FinalMessage | ToolCallEvent] = []
    text = response.output_text
    if text:
        events.append(TextDelta(str(text)))
    calls = []
    for item in response.output or []:
        item_type = item.type
        if item_type == "message":
            for content in item.content or []:
                if content.text:
                    events.append(TextDelta(str(content.text)))
        elif item_type in {"function_call", "tool_call"}:
            calls.append(tool_call_from_response_item(item))
    if calls:
        events.append(ToolCallEvent([ToolCall(**c) for c in calls]))
    else:
        events.append(FinalMessage(str(text or ""), "end_turn"))
    return events
