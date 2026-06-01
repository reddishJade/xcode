from __future__ import annotations

from collections.abc import Callable
from typing import Any


class HookManager:
    """类型安全的 hook 注册/触发管理器。

    支持 before/after 命名约定的 hook 事件。
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def register(self, event: str, handler: Callable[..., Any]) -> None:
        """注册 hook handler。"""
        self._handlers.setdefault(event, []).append(handler)

    def remove(self, event: str, handler: Callable[..., Any]) -> None:
        """移除 hook handler。"""
        handlers = self._handlers.get(event)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            del self._handlers[event]

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
