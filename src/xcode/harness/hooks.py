from __future__ import annotations

from collections.abc import Callable
from typing import Any


class HookManager:
    """类型安全的 hook 注册/触发管理器。

    支持 before/after 命名约定的 hook 事件。
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def on(self, event: str, handler: Callable[..., Any]) -> Callable[[], None]:
        """注册 hook handler。返回取消注册函数。"""
        self._handlers.setdefault(event, []).append(handler)
        return lambda: self._handlers[event].remove(handler)

    async def emit(self, event: str, **kwargs: Any) -> list[Any]:
        """顺序触发所有 handler，返回结果列表。"""
        results: list[Any] = []
        for handler in self._handlers.get(event, []):
            result = handler(**kwargs)
            if hasattr(result, "__await__"):
                result = await result
            results.append(result)
        return results

    def clear(self) -> None:
        self._handlers.clear()
