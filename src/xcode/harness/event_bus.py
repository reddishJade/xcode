from __future__ import annotations

from collections.abc import Callable
from typing import Any


class EventEmitter:
    """简单的事件总线。"""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., None]]] = {}

    def register(self, channel: str, handler: Callable[..., None]) -> None:
        """注册事件处理器。"""
        self._handlers.setdefault(channel, []).append(handler)

    def remove(self, channel: str, handler: Callable[..., None]) -> None:
        """移除事件处理器。"""
        handlers = self._handlers.get(channel)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            del self._handlers[channel]

    def emit(self, channel: str, **data: Any) -> None:
        """触发事件。"""
        for handler in self._handlers.get(channel, []):
            handler(**data)

    def clear(self) -> None:
        self._handlers.clear()
