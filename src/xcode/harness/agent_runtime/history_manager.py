"""会话历史管理。

负责消息存储、加载、RunState 序列化/反序列化。
"""

from __future__ import annotations

import json
from typing import Any, cast

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


class HistoryManager:
    """会话历史存储与恢复。"""

    def __init__(self) -> None:
        self._history: list[AgentMessage] = []

    def clear(self) -> list[AgentMessage]:
        """清空历史，返回被清空的消息。"""
        old = self._history
        self._history = []
        return old

    def load(self, messages: list[AgentMessage]) -> None:
        """用外部消息替换当前历史。"""
        from copy import deepcopy

        self._history = deepcopy(messages)

    def load_run_state(self, run_state: RunState) -> None:
        """从可序列化运行状态恢复消息。"""
        self._history = _messages_from_run_state(run_state)

    def messages(self) -> list[AgentMessage]:
        """返回当前历史的浅拷贝。"""
        return list(self._history)

    def save_turn(self, messages: list[AgentMessage]) -> None:
        """保存本轮消息到历史。"""
        self._history.extend(messages)

    @property
    def current_mode_from_state(self) -> str | None:
        """从 RunState 恢复时可能附带的模式信息。"""
        return getattr(self, "_restored_mode", None)

    def restore_mode(self, run_state: RunState) -> str | None:
        """从 RunState 提取模式信息。"""
        mode = run_state.current_mode
        if mode in {"act", "plan", "review"}:
            return mode
        return None


# ── RunState 反序列化 ──


def _messages_from_run_state(run_state: RunState) -> list[AgentMessage]:
    """从可序列化运行状态恢复模型可见消息。"""
    messages: list[AgentMessage] = []
    for item in run_state.messages:
        message = _message_from_dict(item)
        if message is not None:
            messages.append(message)
    return messages


def _message_from_dict(item: dict[str, Any]) -> AgentMessage | None:
    role = str(item.get("role", ""))
    if role == "system":
        return SystemMessage(content=str(item.get("content", "")))
    if role == "user":
        return UserMessage(content=str(item.get("content", "")))
    if role == "assistant":
        return _assistant_from_dict(item)
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=str(item.get("tool_call_id", "")),
            content=str(item.get("content", "")),
        )
    return None


def _assistant_from_dict(item: dict[str, Any]) -> AssistantMessage:
    content: list[ContentBlock] = []
    text = item.get("content")
    if isinstance(text, str) and text:
        content.append(TextContent(text=text))
    tool_calls = item.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            parsed = _tool_call_from_dict(tool_call)
            if parsed is not None:
                content.append(parsed)
    return AssistantMessage(content=content)


def _tool_call_from_dict(item: object) -> ToolCallContent | None:
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
