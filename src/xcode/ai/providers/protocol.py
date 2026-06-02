from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from xcode.ai.events import Message, ProviderEvent
from xcode.ai.types import ToolDefinition


class ModelProvider(Protocol):
    """Provider 的最小流式协议。"""

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[ProviderEvent]: ...
