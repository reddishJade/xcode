from __future__ import annotations

from collections.abc import Callable
from typing import Any


class EventEmitter:
    """简单的发布/订阅事件总线。"""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., None]]] = {}

    def on(self, channel: str, handler: Callable[..., None]) -> Callable[[], None]:
        """注册事件处理器，返回取消注册函数。"""
        self._handlers.setdefault(channel, []).append(handler)
        return lambda: self._handlers[channel].remove(handler)

    def emit(self, channel: str, **data: Any) -> None:
        """触发事件。"""
        for handler in self._handlers.get(channel, []):
            handler(**data)

    def clear(self) -> None:
        self._handlers.clear()
