"""OpenAI tool schema and message format codecs.

Schema 转换、消息格式转换（Chat Completions / Responses API）。
流式事件解码见 stream_codec.py。
"""

from __future__ import annotations

import copy
import logging

import orjson
from typing import Any, Protocol


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
    """将 JSON Schema 转换为 OpenAI strict mode 兼容格式。

    OpenAI strict mode 约束：
    - object 类型必须声明所有字段为 required
    - 不允许 additionalProperties（必须显式禁止）
    - 不支持约束字段（minLength/maxLength/minItems/maxItems）

    这些限制确保模型生成的 JSON 严格匹配 schema，避免幻觉字段。
    """
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

        # 移除 OpenAI strict mode 不支持的约束字段
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
                "name": name,
                "description": description,
                "parameters": resolved,
                "strict": True,
            },
        }
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": resolved},
    }


_KNOWN_BUILTIN_TOOL_TYPES = frozenset(
    {
        "web_search_preview",
        "web_search",
        "file_search",
        "code_interpreter",
        "shell",
        "computer_use_preview",
    }
)


def to_responses_tool(
    name: str,
    description: str,
    schema: dict | None,
    builtin: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """将工具定义转换为 Responses API 扁平格式。

    Responses API 中 function tool 的 name/description/parameters 为顶层字段，
    且支持 strict 模式保证输出 schema 一致性。

    builtin 工具（如 web_search、shell）直接透传，
    但必须包含非空 type 字段且值为已知内置类型。
    """
    if builtin is not None:
        tool_type = builtin.get("type")
        if not isinstance(tool_type, str) or not tool_type:
            raise ValueError(
                "builtin tool definition must have a non-empty 'type' field, "
                f"got: {builtin}"
            )
        if tool_type not in _KNOWN_BUILTIN_TOOL_TYPES:
            logging.getLogger(__name__).warning(
                "builtin tool type=%r is not in the known types list; "
                "may not be supported by the current API version",
                tool_type,
            )
        return dict(builtin)
    resolved = schema or {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": description},
        },
    }
    if strict:
        resolved = make_schema_strict(resolved)
    result: dict[str, Any] = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": resolved,
    }
    if strict:
        result["strict"] = True
    return result


# Provider 之间无需转换的目标列表（共享 reasoning_content 协议）
_REASONING_CONTENT_TRANSPORTS = {"deepseek_chat", "chatglm_chat", "mimo_chat"}

# 通过 reasoning_content 字段流式传输思考内容的 provider 列表
# （来自这些 provider 的消息保留 reasoning_content；发往非这些 provider 时需转为文本）


def _has_reasoning_content(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("reasoning_content"):
            return True
    return False


def normalize_cross_provider_messages(
    messages: list[dict[str, Any]],
    target_transport: str,
) -> list[dict[str, Any]]:
    if not _has_reasoning_content(messages):
        return messages

    if target_transport in _REASONING_CONTENT_TRANSPORTS:
        return messages

    result: list[dict[str, Any]] = []
    for msg in messages:
        rc = msg.get("reasoning_content")
        if rc:
            msg = copy.deepcopy(msg)
            text = str(rc)
            existing = msg.get("content")
            if existing is None or existing == "":
                msg["content"] = f"<thinking>{text}</thinking>"
            elif isinstance(existing, str):
                msg["content"] = f"<thinking>{text}</thinking>\n\n{existing}"
            msg.pop("reasoning_content", None)
        result.append(msg)
    return result


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


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """转换为 Responses API 格式。tool result 使用 type:"function_call_output"。"""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")

        if role == "tool" and "tool_call_id" in message:
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": str(content) if content is not None else "",
                }
            )
            continue

        if role == "assistant" and "tool_calls" in message:
            text_content = str(content) if content else None
            if text_content:
                converted.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                    }
                )
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
                fixed_function["arguments"] = orjson.dumps(arguments).decode()
            fixed_call["function"] = fixed_function
        normalized.append(fixed_call)
    return normalized


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
                        "arguments": orjson.dumps(
                            part.get("input", {}),
                        ).decode(),
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
    role: str, content: list
) -> list[dict[str, Any]]:
    """将内容块转换为 Responses API 格式。"""
    converted: list[dict[str, Any]] = []
    function_calls = []
    message_content: list[dict[str, Any]] = []

    for part in content:
        if not isinstance(part, dict) and not hasattr(part, "type"):
            message_content.append(_responses_text_part(role, str(part)))
            continue
        part_type = str(_part_value(part, "type", ""))
        if part_type == "tool_result":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": str(_part_value(part, "tool_use_id", "")),
                    "output": str(_part_value(part, "content", "")),
                }
            )
        elif part_type == "tool_use":
            function_calls.append(
                {
                    "type": "function_call",
                    "call_id": str(_part_value(part, "id", "")),
                    "name": str(_part_value(part, "name", "")),
                    "arguments": orjson.dumps(_part_value(part, "input", {})).decode(),
                }
            )
        elif part_type == "text":
            text = str(_part_value(part, "text", ""))
            message_content.append(_responses_text_part(role, text))
        elif part_type in {"image", "image_url", "input_image"}:
            message_content.append(_responses_image_part(part))
        elif part_type in {"file", "input_file"}:
            message_content.append(_responses_file_part(part))

    if message_content:
        converted.append(
            {
                "type": "message",
                "role": role,
                "content": message_content,
            }
        )

    converted.extend(function_calls)
    return converted


def _part_value(part: object, key: str, default: Any = None) -> Any:
    """从 dict 或内容块对象读取字段。"""
    if isinstance(part, dict):
        return part.get(key, default)
    return getattr(part, key, default)


def _responses_text_part(role: str, text: str) -> dict[str, Any]:
    """构造 Responses 文本内容块。"""
    if role == "assistant":
        return {"type": "output_text", "text": text}
    return {"type": "input_text", "text": text}


def _responses_image_part(part: object) -> dict[str, Any]:
    """构造 Responses 图像输入块。"""
    source = _part_value(part, "source", None)
    if isinstance(source, dict):
        result = {"type": "input_image", **source}
    else:
        result = {
            "type": "input_image",
            "image_url": _part_value(part, "image_url", ""),
        }
    if "url" in result and "image_url" not in result:
        result["image_url"] = result.pop("url")
    return result


def _responses_file_part(part: object) -> dict[str, Any]:
    """构造 Responses 文件输入块。"""
    source = _part_value(part, "source", None)
    if isinstance(source, dict):
        return {"type": "input_file", **source}
    return {
        "type": "input_file",
        "file_id": _part_value(part, "file_id", ""),
    }
