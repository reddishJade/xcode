from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition


type RouterFn = Callable[
    [list[dict[str, Any]], list[ToolDefinition]], str
]
"""路由函数：根据消息和工具定义，返回 provider 名称以选择目标 provider。"""


class RouterProvider:
    """根据路由函数在多个 provider 之间动态切换。

    支持两种模式：
    1. pre-route: 每次 stream() 调用前通过 RouterFn 选择 provider
    2. fallback: 主 provider 失败时降级到备 provider
    """

    def __init__(
        self,
        providers: dict[str, Any],
        router: RouterFn | None = None,
        default: str = "",
        fallback: str | None = None,
    ) -> None:
        self._providers = providers
        self._router = router
        self._fallback = fallback
        self._default = default or next(iter(providers.keys()))
        self._last_provider: str = self._default

    @property
    def model(self) -> str:
        provider = self._providers.get(self._last_provider)
        return getattr(provider, "model", "unknown")

    @property
    def active_provider(self) -> Any:
        return self._providers.get(self._last_provider)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        name = self._default
        if self._router:
            name = self._router(messages, tools) or self._default
        provider = self._providers.get(name)
        if provider is None:
            provider = self._providers.get(self._default)
        if provider is None:
            return

        self._last_provider = name
        try:
            async for event in provider.stream(messages, tools, options=options, **kwargs):
                yield event
        except Exception:
            if self._fallback and self._fallback != name:
                fb = self._providers.get(self._fallback)
                if fb is not None:
                    self._last_provider = self._fallback
                    async for event in fb.stream(messages, tools, options=options, **kwargs):
                        yield event
                    return
            raise
