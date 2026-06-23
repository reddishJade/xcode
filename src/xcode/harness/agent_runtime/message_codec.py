"""Compacted dict ↔ AgentMessage 编解码。"""

from __future__ import annotations

import json
from typing import Any

from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from ...agent.protocols import ContentBlock
from ...agent.types import TextContent, ToolCallContent
from .result import RunState


def messages_from_compacted_dicts(
    messages: list[dict[str, Any]],
) -> list[AgentMessage]:
    """将压缩后的 provider 风格消息恢复为内部消息。"""
    restored: list[AgentMessage] = []
    for item in messages:
        message = _message_from_compacted_dict(item)
        if message is not None:
            restored.append(message)
    return restored


def messages_from_run_state(run_state: RunState) -> list[AgentMessage]:
    """从可序列化运行状态恢复模型可见消息。"""
    return messages_from_compacted_dicts(run_state.messages)


def _message_from_compacted_dict(item: dict[str, Any]) -> AgentMessage | None:
    """恢复单条压缩消息，未知格式保持为普通用户文本。"""
    role = str(item.get("role", ""))
    content = item.get("content", "")
    if role == "system":
        return SystemMessage(content=_content_to_text(content))
    if role == "user":
        return UserMessage(content=_content_to_text(content))
    if role == "assistant":
        return _assistant_from_compacted_dict(item)
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=str(item.get("tool_call_id", "")),
            content=_tool_result_content_from_compacted(content),
        )
    return None


def _assistant_from_compacted_dict(item: dict[str, Any]) -> AssistantMessage:
    """恢复 assistant 文本和工具调用。"""
    content_blocks: list[ContentBlock] = []
    content = item.get("content")
    text = _content_to_text(content)
    if text:
        content_blocks.append(TextContent(text=text))

    tool_calls = item.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            parsed = _tool_call_from_compacted(tool_call)
            if parsed is not None:
                content_blocks.append(parsed)
    return AssistantMessage(content=content_blocks)


def _tool_call_from_compacted(item: object) -> ToolCallContent | None:
    """恢复 OpenAI 风格 tool_call。"""
    if not isinstance(item, dict):
        return None
    function = item.get("function", {})
    if not isinstance(function, dict):
        return None
    name = str(function.get("name", "")).strip()
    tool_call_id = str(item.get("id", "")).strip()
    if not name or not tool_call_id:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = {}
        arguments = decoded
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCallContent(id=tool_call_id, name=name, arguments=arguments)


def _tool_result_content_from_compacted(content: object) -> str:
    """恢复工具结果文本。"""
    if not isinstance(content, list):
        return _content_to_text(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "tool_result":
            parts.append(str(part.get("content", "")))
        else:
            parts.append(_content_to_text(part))
    return "".join(parts)


def _content_to_text(content: object) -> str:
    """将压缩中间格式转为稳定文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "tool_result":
                parts.append(str(part.get("content", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)
