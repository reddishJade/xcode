from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from xcode.ai.events import Message, ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition


class ModelProvider(Protocol):
    """Provider 的最小流式协议。"""

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]: ...
