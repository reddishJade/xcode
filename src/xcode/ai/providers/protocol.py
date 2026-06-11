from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from xcode.ai.events import Message, ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition


class ModelProvider(Protocol):
    """Provider 的流式调用与运行时元数据协议。"""

    @property
    def model(self) -> str: ...

    @property
    def thinking(self) -> bool: ...

    @property
    def reasoning_effort(self) -> str | None: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]: ...
