from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
import queue
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
        self._submit_queue: queue.Queue[
            tuple[Coroutine[Any, Any, Any], concurrent.futures.Future[Any]] | None
        ] = queue.Queue()
        self._tasks: set[asyncio.Task[Any]] = set()

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

        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        self._submit_queue.put((coro, future))
        return future

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._submit_queue.put(None)
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
            loop.run_until_complete(self._dispatcher())
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

    async def _dispatcher(self) -> None:
        """持续从线程安全队列读取提交的协程并在本 loop 内执行。"""
        while True:
            try:
                item = self._submit_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if item is None:
                break
            coro, future = item
            if future.cancelled():
                coro.close()
                continue
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(
                lambda done, target=future: self._complete_future(done, target)
            )
        if self._tasks:
            for task in tuple(self._tasks):
                task.cancel()
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def _complete_future(
        self,
        task: asyncio.Task[Any],
        future: concurrent.futures.Future[Any],
    ) -> None:
        """将 loop 内 task 的完成状态映射回线程安全 Future。"""
        self._tasks.discard(task)
        if future.cancelled():
            return
        if task.cancelled():
            future.cancel()
            return
        exc = task.exception()
        if exc is not None:
            future.set_exception(exc)
            return
        future.set_result(task.result())
