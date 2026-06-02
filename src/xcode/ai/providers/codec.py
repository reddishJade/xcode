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

"""OpenAI tool schema and stream delta codecs consolidated."""


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


class _ChatToolCallFunction(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> str: ...


class _ChatToolCall(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def function(self) -> _ChatToolCallFunction: ...


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


def make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    import copy

    s = copy.deepcopy(schema)

    def process(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        t = node.get("type")
        if t == "object" or "properties" in node:
            properties = node.get("properties", {})
            if properties:
                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

        node.pop("minLength", None)
        node.pop("maxLength", None)
        node.pop("minItems", None)
        node.pop("maxItems", None)

        if "properties" in node:
            node["properties"] = {k: process(v) for k, v in node["properties"].items()}
        if "items" in node:
            node["items"] = process(node["items"])
        if "anyOf" in node:
            node["anyOf"] = [process(item) for item in node["anyOf"]]
        if "allOf" in node:
            node["allOf"] = [process(item) for item in node["allOf"]]
        if "oneOf" in node:
            node["oneOf"] = [process(item) for item in node["oneOf"]]

        return node

    return process(s)


def to_chat_tool(
    name: str, description: str, schema: dict | None, strict: bool = False
) -> dict[str, Any]:
    resolved = schema or {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": description,
            }
        },
    }
    if strict:
        resolved = make_schema_strict(resolved)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": resolved,
                "strict": True,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": resolved,
        },
    }


def to_responses_tool(
    name: str, description: str, schema: dict | None
) -> dict[str, Any]:
    resolved = schema or {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": description,
            }
        },
    }
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": resolved,
    }


def to_chat_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """转换为 Chat Completions API 格式。tool result 使用 role:"tool"。"""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if "content" in message:
            if role == "tool" and content is None:
                result_content: str | None = ""
            else:
                result_content = None if content is None else str(content)
        else:
            result_content = ""

        if isinstance(content, list):
            converted.extend(
                _content_blocks_to_chat_messages(
                    role,
                    content,
                    reasoning_content=message.get("reasoning_content"),
                    prefix=message.get("prefix"),
                )
            )
        else:
            result: dict[str, Any] = {"role": role, "content": result_content}
            if "tool_calls" in message:
                result["tool_calls"] = _normalize_chat_tool_calls(message["tool_calls"])
            if (
                "reasoning_content" in message
                and message["reasoning_content"] is not None
            ):
                result["reasoning_content"] = message["reasoning_content"]
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "prefix" in message:
                result["prefix"] = message["prefix"]
            converted.append(result)
    return converted


def to_responses_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """转换为 Responses API 格式。tool result 使用 type:"function_call_output"。"""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")

        # tool result 转换为 function_call_output
        if role == "tool" and "tool_call_id" in message:
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": str(content) if content is not None else "",
                }
            )
            continue

        # assistant 消息带 tool_calls
        if role == "assistant" and "tool_calls" in message:
            # 先添加文本内容
            text_content = str(content) if content else None
            if text_content:
                converted.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                    }
                )
            # 再添加 tool_calls
            for call in message["tool_calls"]:
                if isinstance(call, dict):
                    func = call.get("function", {})
                    converted.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        }
                    )
            continue

        # 普通消息
        if isinstance(content, list):
            converted.extend(_content_blocks_to_responses_input(role, content))
        else:
            text = str(content) if content is not None else ""
            if role == "assistant":
                converted.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            else:
                converted.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    }
                )

    return converted


def _normalize_chat_tool_calls(tool_calls: object) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return normalized
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fixed_call = dict(call)
        function = fixed_call.get("function")
        if isinstance(function, dict):
            fixed_function = dict(function)
            arguments = fixed_function.get("arguments", "")
            if not isinstance(arguments, str):
                fixed_function["arguments"] = json.dumps(
                    arguments,
                    ensure_ascii=False,
                )
            fixed_call["function"] = fixed_function
        normalized.append(fixed_call)
    return normalized


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        result = json.loads(raw_arguments or "{}")
        return result if isinstance(result, dict) else {"input": str(result)}
    except json.JSONDecodeError:
        return {"input": raw_arguments}


def _content_blocks_to_chat_messages(
    role: str,
    content: list,
    reasoning_content: str | None = None,
    prefix: bool | None = None,
) -> list[dict[str, Any]]:
    """将内容块转换为 Chat Completions 格式。"""
    converted: list[dict[str, Any]] = []
    assistant_tool_calls = []
    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        if part.get("type") == "tool_result":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": str(part.get("tool_use_id", "")),
                    "content": str(part.get("content", "")),
                }
            )
        elif part.get("type") == "tool_use":
            assistant_tool_calls.append(
                {
                    "id": str(part.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(part.get("name", "")),
                        "arguments": json.dumps(
                            part.get("input", {}),
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        elif part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
    if assistant_tool_calls:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": assistant_tool_calls,
        }
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if prefix is not None:
            msg["prefix"] = prefix
        converted.insert(0, msg)
    elif text_parts:
        msg = {"role": role, "content": "".join(text_parts)}
        if reasoning_content is not None and role == "assistant":
            msg["reasoning_content"] = reasoning_content
        if prefix is not None and role == "assistant":
            msg["prefix"] = prefix
        converted.insert(0, msg)
    return converted


def _content_blocks_to_responses_input(
    role: str,
    content: list,
) -> list[dict[str, Any]]:
    """将内容块转换为 Responses API 格式。"""
    converted: list[dict[str, Any]] = []
    function_calls = []
    text_parts = []

    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue

        part_type = part.get("type")
        if part_type == "tool_result":
            # tool result -> function_call_output
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": str(part.get("tool_use_id", "")),
                    "output": str(part.get("content", "")),
                }
            )
        elif part_type == "tool_use":
            # tool_use -> function_call
            function_calls.append(
                {
                    "type": "function_call",
                    "call_id": str(part.get("id", "")),
                    "name": str(part.get("name", "")),
                    "arguments": json.dumps(part.get("input", {}), ensure_ascii=False),
                }
            )
        elif part_type == "text":
            text_parts.append(str(part.get("text", "")))

    # 添加文本消息
    if text_parts:
        text = "".join(text_parts)
        if role == "assistant":
            converted.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        else:
            converted.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": text}],
                }
            )

    # 添加 function_call
    converted.extend(function_calls)

    return converted


def tool_call_from_chat(call: _ChatToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.function.name,
        "input": parse_tool_arguments(call.function.arguments),
    }


def tool_call_from_response_item(item: _ResponseOutputItem) -> dict[str, Any]:
    return {
        "id": str(item.call_id or item.id or ""),
        "name": str(item.name or ""),
        "input": parse_tool_arguments(str(item.arguments or "{}")),
    }


def chat_stream_to_events(
    stream: Iterable[_ChatCompletionChunk],
) -> Iterator[TextDelta | ReasoningDelta | ToolCallEvent | UsageUpdate]:
    """Yields provider events dicts: TextDelta, ReasoningDelta, ToolCallEvent, UsageUpdate."""
    calls: dict[int, dict[str, str]] = defaultdict(
        lambda: {"id": "", "name": "", "arguments": ""}
    )
    for chunk in stream:
        usage = chunk.usage
        if usage is not None:
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            yield UsageUpdate(input_tokens, output_tokens)

        choices = chunk.choices
        if not choices:
            continue
        choice = choices[0]
        delta = choice.delta
        reasoning = getattr(
            delta, "reasoning_content", None
        )  # DeepSeek 非标准扩展字段，部分模型不返回
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
        ToolCall(
            id=current["id"],
            name=current["name"],
            input=parse_tool_arguments(current["arguments"]),
        )
        for index, current in sorted(calls.items())
    ]
    if ready:
        yield ToolCallEvent(ready)


def responses_stream_to_events(
    stream: Iterable[_ResponseStreamEvent],
) -> Iterator[TextDelta | ReasoningDelta | ToolCallEvent | UsageUpdate | FinalMessage]:
    """处理 Responses API 流式事件。

    支持的事件类型：
    - response.output_text.delta: 文本增量
    - response.function_call_arguments.delta: 工具调用参数增量
    - response.function_call_arguments.done: 工具调用参数完成
    - response.output_item.done: 输出项完成（包含 function_call）
    - response.completed: 响应完成（包含 usage 和 FinalMessage）

    只有在收到 response.completed 后才会 yield FinalMessage，
    避免异常断流被伪装成正常 end_turn。
    """
    # 累积工具调用和文本
    pending_calls: dict[int, dict[str, str]] = {}
    accumulated_text = ""
    completed = False
    response_text = ""  # 从 completed 响应中提取的完整 output_text

    for event in stream:
        event_type = event.type

        # 文本增量
        if event_type == "response.output_text.delta":
            text = event.delta
            if text:
                accumulated_text += str(text)
                yield TextDelta(str(text))

        # reasoning 增量（部分模型支持）
        elif event_type == "response.reasoning_summary_text.delta":
            text = event.delta
            if text:
                yield ReasoningDelta(str(text))

        # 工具调用参数增量
        elif event_type == "response.function_call_arguments.delta":
            # 从 event 中提取 index 和 delta
            index = getattr(event, "output_index", 0)
            delta = getattr(event, "delta", "")
            if index not in pending_calls:
                pending_calls[index] = {"id": "", "name": "", "arguments": ""}
            pending_calls[index]["arguments"] += delta

        # 输出项完成
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

        # 响应完成
        elif event_type == "response.completed":
            completed = True
            response = getattr(event, "response", None)
            if response is not None:
                # 捕获 output_text，优先用于 FinalMessage
                response_text = str(getattr(response, "output_text", "") or "")
                # 提取 usage
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0) or getattr(
                        usage, "prompt_tokens", 0
                    )
                    output_tokens = getattr(usage, "output_tokens", 0) or getattr(
                        usage, "completion_tokens", 0
                    )
                    yield UsageUpdate(int(input_tokens), int(output_tokens))

    # 输出累积的工具调用
    if pending_calls:
        ready = [
            ToolCall(
                id=call["id"],
                name=call["name"],
                input=parse_tool_arguments(call["arguments"]),
            )
            for _, call in sorted(pending_calls.items())
        ]
        yield ToolCallEvent(ready)
    elif completed:
        # 优先使用 response.output_text，回退到 accumulated_text
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
                content_text = content.text
                if content_text:
                    events.append(TextDelta(str(content_text)))
        elif item_type in {"function_call", "tool_call"}:
            calls.append(tool_call_from_response_item(item))
    if calls:
        events.append(ToolCallEvent([ToolCall(**c) for c in calls]))
    else:
        events.append(FinalMessage(str(text or ""), "end_turn"))
    return events
