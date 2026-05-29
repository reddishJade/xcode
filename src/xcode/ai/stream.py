from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Generic, TypeVar

"""EventStream：AI 层异步事件推送与收集。"""

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class StreamEvent[T]:
    type: str
    data: T | None = None


class EventStream(Generic[T, R]):
    """异步事件流封装。

    基于 asyncio.Queue，push() 在任意线程安全投放事件，
    __aiter__ 在 event loop 中消费。end() 时通过 collected
    参数决定是否返回收集值。
    """

    def __init__(
        self,
        is_end: type | tuple[type, ...] | None = None,
        collected: type | None = None,
    ) -> None:
        self._queue: asyncio.Queue[StreamEvent[T] | Exception] = asyncio.Queue()
        self._is_end = is_end or ()
        self._collected = collected
        self._result: R | None = None

    def push(self, event: T) -> None:
        self._queue.put_nowait(StreamEvent(type="data", data=event))

    def end(self, result: R | None = None) -> None:
        self._result = result
        self._queue.put_nowait(StreamEvent(type="end"))

    def _is_end_event(self, event: T) -> bool:
        if isinstance(event, type):
            return isinstance(event, self._is_end)  # type: ignore
        return False

    async def result(self) -> R | None:
        return self._result

    def __aiter__(self) -> AsyncIterator[StreamEvent[T]]:
        return self._async_gen()

    async def _async_gen(self) -> AsyncIterator[StreamEvent[T]]:

        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=None)
            except asyncio.CancelledError:
                return
            if isinstance(event, Exception):
                raise event
            if event.type == "end":
                return
            yield event
