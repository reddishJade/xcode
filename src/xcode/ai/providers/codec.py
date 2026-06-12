"""OpenAI tool schema and message format codecs.

Schema 转换、消息格式转换（Chat Completions）、
工具列表规范化与指纹计算。
流式事件解码见 stream_codec.py。
"""

from __future__ import annotations

import copy
import hashlib
import json

import orjson
from typing import Any, Protocol

from xcode.ai.types import ToolDefinition


_UNSUPPORTED_STRICT_SCHEMA_KEYS = frozenset(
    {
        "default",
        "format",
        "maximum",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minimum",
        "minItems",
        "minLength",
        "minProperties",
        "multipleOf",
        "pattern",
        "patternProperties",
        "propertyNames",
        "uniqueItems",
    }
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
    """将 JSON Schema 转换为 OpenAI strict mode 兼容格式。

    OpenAI strict mode 约束：
    - object 类型必须声明所有字段为 required
    - 原本可选字段用 null union 表达
    - 不允许 additionalProperties（必须显式禁止）
    - 不支持细粒度校验约束字段

    这些限制确保模型生成的 JSON 严格匹配 schema，避免幻觉字段。
    """
    s = copy.deepcopy(schema)

    def process(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        t = node.get("type")
        if t == "object" or "properties" in node:
            properties = node.get("properties", {})
            existing_required = node.get("required", [])
            required = (
                set(existing_required) if isinstance(existing_required, list) else set()
            )
            if isinstance(properties, dict) and properties:
                node["properties"] = {
                    key: _nullable_schema(value) if key not in required else value
                    for key, value in properties.items()
                }
                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

        for key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
            node.pop(key, None)

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


def _nullable_schema(schema: Any) -> Any:
    """把原可选字段编码为 strict schema 可接受的 null union。"""
    if not isinstance(schema, dict):
        return schema
    result = copy.deepcopy(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            result["type"] = [*schema_type, "null"]
        return result
    if isinstance(schema_type, str):
        if schema_type != "null":
            result["type"] = [schema_type, "null"]
        return result
    if "anyOf" in result:
        any_of = result.get("anyOf")
        if isinstance(any_of, list):
            has_null = any(
                isinstance(item, dict) and item.get("type") == "null" for item in any_of
            )
            if not has_null:
                result["anyOf"] = [*any_of, {"type": "null"}]
    return result


def to_chat_tool(
    name: str, description: str, schema: dict | None, strict: bool = False
) -> dict[str, Any]:
    """将工具定义转换为 Chat Completions 格式。"""
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


def to_chat_tools(
    tools: tuple[ToolDefinition, ...],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """将 ToolDefinition 列表转换为 Chat Completions 工具参数。"""
    return [
        to_chat_tool(
            tool.name,
            tool.description,
            tool.parameters,
            strict=strict,
        )
        for tool in tools
    ]


# Provider 之间无需转换的目标列表（共享 reasoning_content 协议）
_REASONING_CONTENT_TRANSPORTS = {"deepseek_chat", "chatglm_chat", "mimo_chat"}


def _has_reasoning_content(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("reasoning_content"):
            return True
    return False


def normalize_cross_provider_messages(
    messages: list[dict[str, Any]],
    target_transport: str,
) -> list[dict[str, Any]]:
    """跨 provider 消息归一化。

    当消息来自不同 provider 时（如 DeepSeek → MiMo），
    将 provider 专有字段（如 reasoning_content）转为通用文本格式。
    """
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


# ── 工具列表规范化与指纹计算 ──


def canonical_tool_schema(tool: ToolDefinition) -> dict[str, Any]:
    """规范化工具 schema（字典键排序）。

    确保同一工具的 schema 在不同调用间字节稳定。
    """
    result: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "schema": tool.parameters,
    }
    if tool.builtin is not None:
        result["builtin"] = tool.builtin
    return _sort_dict_recursive(result)


def _sort_dict_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_dict_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_dict_recursive(item) for item in obj]
    return obj


def canonical_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """按 name 排序并规范化工具列表。

    确保工具列表在不同调用间顺序稳定。
    """
    sorted_tools = sorted(tools, key=lambda t: t.name)
    return [canonical_tool_schema(t) for t in sorted_tools]


def tool_catalog_fingerprint(tools: list[ToolDefinition]) -> str:
    """计算工具集合指纹（SHA256）。

    用于检测工具 catalog 漂移。
    """
    canonical = canonical_tools(tools)
    serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
