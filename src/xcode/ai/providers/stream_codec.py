"""流式事件解码。

处理 Chat Completions 和 Responses API 的流式响应，
将原始 chunk 转换为统一的 ProviderEvent。
"""

from __future__ import annotations

import orjson
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
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
    def tool_calls(self) -> Sequence[_ChoiceDeltaToolCall] | None: ...


class _Choice(Protocol):
    @property
    def delta(self) -> _ChoiceDelta: ...


class _ChatCompletionChunk(Protocol):
    @property
    def usage(self) -> _Usage | None: ...
    @property
    def choices(self) -> Sequence[_Choice]: ...


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
    @property
    def action(self) -> Any | None: ...


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
        result = orjson.loads((raw_arguments or "{}").encode())
        return result if isinstance(result, dict) else {"input": str(result)}
    except orjson.JSONDecodeError:
        return {"input": raw_arguments}


def tool_call_from_response_item(item: _ResponseOutputItem) -> dict[str, Any]:
    return {
        "id": str(item.call_id or item.id or ""),
        "name": str(item.name or ""),
        "input": parse_tool_arguments(str(item.arguments or "{}")),
    }


def shell_call_from_response_item(item: _ResponseOutputItem) -> ToolCall:
    """将 Responses shell_call item 转换为内部工具调用。"""
    return ToolCall(
        id=str(item.call_id or item.id or ""),
        name="shell",
        input=_shell_action_input(item.action),
    )


def _shell_action_input(action: Any | None) -> dict[str, Any]:
    """保留官方 shell action 字段，供后续执行层处理。"""
    if action is None:
        return {}
    commands = _action_value(action, "commands")
    timeout_ms = _action_value(action, "timeout_ms")
    max_output_length = _action_value(action, "max_output_length")

    result: dict[str, Any] = {}
    if isinstance(commands, list):
        result["commands"] = [str(command) for command in commands]
    elif commands is not None:
        result["commands"] = [str(commands)]
    if timeout_ms is not None:
        result["timeout_ms"] = timeout_ms
    if max_output_length is not None:
        result["max_output_length"] = max_output_length
    return result


def _action_value(action: Any, key: str) -> Any | None:
    """从 SDK action 对象或 dict 中读取字段。"""
    if isinstance(action, dict):
        return action.get(key)
    return getattr(action, key, None)


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
    pending_shell_calls: dict[int, ToolCall] = {}
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
        elif _is_reasoning_delta_event(event_type):
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
                elif item_type == "shell_call":
                    index = getattr(event, "output_index", 0)
                    pending_shell_calls[index] = shell_call_from_response_item(item)
        elif event_type == "response.completed":
            completed = True
            response = getattr(event, "response", None)
            if response is not None:
                response_text = str(getattr(response, "output_text", "") or "")
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0) or getattr(
                        usage, "prompt_tokens", 0
                    )
                    output_tokens = getattr(usage, "output_tokens", 0) or getattr(
                        usage, "completion_tokens", 0
                    )
                    yield UsageUpdate(int(input_tokens), int(output_tokens))

    if pending_calls or pending_shell_calls:
        ready_by_index = {
            index: ToolCall(
                id=call["id"],
                name=call["name"],
                input=parse_tool_arguments(call["arguments"]),
            )
            for index, call in pending_calls.items()
        }
        ready_by_index.update(pending_shell_calls)
        ready = [call for _, call in sorted(ready_by_index.items())]
        yield ToolCallEvent(ready)
    elif completed:
        final_text = response_text or accumulated_text
        yield FinalMessage(final_text, "end_turn")


def _is_reasoning_delta_event(event_type: str) -> bool:
    """判断 Responses 事件是否是 reasoning 文本增量。"""
    return (
        event_type == "response.reasoning_summary_text.delta"
        or event_type == "response.reasoning_text.delta"
        or (
            event_type.startswith("response.reasoning")
            and event_type.endswith(".delta")
        )
    )
