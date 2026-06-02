"""OpenAI tool schema and message format codecs.

Schema 转换、消息格式转换（Chat Completions / Responses API）。
流式事件解码见 stream_codec.py。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from xcode.ai.events import ToolCall

# re-export 流式函数，保持公共 API 不变
from .stream_codec import (  # noqa: F401
    chat_stream_to_events,
    parse_tool_arguments,
    responses_stream_to_events,
    responses_to_events,
    tool_call_from_response_item,
)


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
            "input": {"type": "string", "description": description},
        },
    }
    if strict:
        resolved = make_schema_strict(resolved)
        return {
            "type": "function",
            "function": {
                "name": name, "description": description,
                "parameters": resolved, "strict": True,
            },
        }
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": resolved},
    }


def to_responses_tool(
    name: str, description: str, schema: dict | None
) -> dict[str, Any]:
    resolved = schema or {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": description},
        },
    }
    return {
        "type": "function", "name": name, "description": description,
        "parameters": resolved,
    }


def to_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    role, content,
                    reasoning_content=message.get("reasoning_content"),
                    prefix=message.get("prefix"),
                )
            )
        else:
            result: dict[str, Any] = {"role": role, "content": result_content}
            if "tool_calls" in message:
                result["tool_calls"] = _normalize_chat_tool_calls(message["tool_calls"])
            if "reasoning_content" in message and message["reasoning_content"] is not None:
                result["reasoning_content"] = message["reasoning_content"]
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "prefix" in message:
                result["prefix"] = message["prefix"]
            converted.append(result)
    return converted


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """转换为 Responses API 格式。tool result 使用 type:"function_call_output"。"""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")

        if role == "tool" and "tool_call_id" in message:
            converted.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": str(content) if content is not None else "",
            })
            continue

        if role == "assistant" and "tool_calls" in message:
            text_content = str(content) if content else None
            if text_content:
                converted.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text_content}],
                })
            for call in message["tool_calls"]:
                if isinstance(call, dict):
                    func = call.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": call.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    })
            continue

        if isinstance(content, list):
            converted.extend(_content_blocks_to_responses_input(role, content))
        else:
            text = str(content) if content is not None else ""
            if role == "assistant":
                converted.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            else:
                converted.append({
                    "type": "message", "role": role,
                    "content": [{"type": "input_text", "text": text}],
                })
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
                fixed_function["arguments"] = json.dumps(arguments, ensure_ascii=False)
            fixed_call["function"] = fixed_function
        normalized.append(fixed_call)
    return normalized


def _content_blocks_to_chat_messages(
    role: str, content: list,
    reasoning_content: str | None = None, prefix: bool | None = None,
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
            converted.append({
                "role": "tool",
                "tool_call_id": str(part.get("tool_use_id", "")),
                "content": str(part.get("content", "")),
            })
        elif part.get("type") == "tool_use":
            assistant_tool_calls.append({
                "id": str(part.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(part.get("name", "")),
                    "arguments": json.dumps(part.get("input", {}), ensure_ascii=False),
                },
            })
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


def _content_blocks_to_responses_input(role: str, content: list) -> list[dict[str, Any]]:
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
            converted.append({
                "type": "function_call_output",
                "call_id": str(part.get("tool_use_id", "")),
                "output": str(part.get("content", "")),
            })
        elif part_type == "tool_use":
            function_calls.append({
                "type": "function_call",
                "call_id": str(part.get("id", "")),
                "name": str(part.get("name", "")),
                "arguments": json.dumps(part.get("input", {}), ensure_ascii=False),
            })
        elif part_type == "text":
            text_parts.append(str(part.get("text", "")))

    if text_parts:
        text = "".join(text_parts)
        if role == "assistant":
            converted.append({
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            })
        else:
            converted.append({
                "type": "message", "role": role,
                "content": [{"type": "input_text", "text": text}],
            })

    converted.extend(function_calls)
    return converted


def tool_call_from_chat(call: _ChatToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.function.name,
        "input": parse_tool_arguments(call.function.arguments),
    }
