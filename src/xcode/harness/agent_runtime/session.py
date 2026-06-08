"""Agent 运行期会话抽象。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from ...agent.messages import AgentMessage


class AgentSession(Protocol):
    """Agent 会话历史存储协议。"""

    def clear(self) -> None:
        """清空会话历史。"""
        ...

    def load(self, messages: list[AgentMessage]) -> None:
        """替换当前会话历史。"""
        ...

    def messages(self) -> list[AgentMessage]:
        """返回当前会话历史副本。"""
        ...

    def append(self, messages: list[AgentMessage]) -> None:
        """追加一轮会话消息。"""
        ...


@dataclass
class InMemoryAgentSession:
    """默认内存会话实现。"""

    _messages: list[AgentMessage] = field(default_factory=list)

    def clear(self) -> None:
        """清空会话历史。"""
        self._messages.clear()

    def load(self, messages: list[AgentMessage]) -> None:
        """替换当前会话历史。"""
        self._messages = deepcopy(messages)

    def messages(self) -> list[AgentMessage]:
        """返回当前会话历史副本。"""
        return deepcopy(self._messages)

    def append(self, messages: list[AgentMessage]) -> None:
        """追加一轮会话消息。"""
        self._messages.extend(deepcopy(messages))
