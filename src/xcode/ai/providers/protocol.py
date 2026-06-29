from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from xcode.ai.events import Message, ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition


class StreamProvider(Protocol):
    """Provider 的流式调用协议。"""

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]: ...


@runtime_checkable
class ModelProvider(StreamProvider, Protocol):
    """Provider 的流式调用与运行时元数据协议。"""

    @property
    def model(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    @property
    def transport(self) -> str: ...

    @property
    def thinking(self) -> bool: ...

    @property
    def reasoning_effort(self) -> str | None: ...
