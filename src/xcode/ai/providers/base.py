"""Provider 抽象协议与基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from xcode.ai.events import Message, ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.ai.usage import UsageTotals


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
    """带运行时元数据的 provider 协议。"""

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

    @property
    def context_window(self) -> int | None: ...

    @property
    def usage_totals(self) -> UsageTotals: ...

    @property
    def cache_hit_rate(self) -> float | None: ...


class Provider(ABC):
    """所有 provider 应继承的抽象基类。

    提供 model/base_url/transport/thinking/reasoning_effort 属性默认实现，
    子类只需实现 stream()。
    """

    def __init__(self) -> None:
        self._model: str = ""
        self._base_url: str = ""
        self._transport: str = ""
        self._thinking: bool = True
        self._reasoning_effort: str | None = None
        self._context_window: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def transport(self) -> str:
        return self._transport

    @property
    def thinking(self) -> bool:
        return self._thinking

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    @property
    def context_window(self) -> int | None:
        """配置覆盖的上下文窗口；None 表示使用模型注册表默认值。"""
        return self._context_window

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]: ...
