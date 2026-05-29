from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
import sys
import threading
from typing import Any, TypeVar

T = TypeVar("T")


class IsolatedAsyncWorker:
    """在独立线程中持有事件循环。"""

    def __init__(self, *, name: str = "xcode-async-worker") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("worker is closed")
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._thread_main,
            name=self._name,
            daemon=True,
        )
        self._thread.start()
        self._started.wait()

        if self._loop is None:
            raise RuntimeError("worker event loop failed to start")

    def submit(
        self,
        coro: Coroutine[Any, Any, T],
    ) -> concurrent.futures.Future[T]:
        if self._closed:
            coro.close()
            raise RuntimeError("worker is closed")

        self.start()

        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _thread_main(self) -> None:
        if sys.platform == "win32":
            loop = asyncio.SelectorEventLoop()
        else:
            loop = asyncio.new_event_loop()

        self._loop = loop
        asyncio.set_event_loop(loop)
        self._started.set()

        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
