"""Benchmark 终端进度展示与非交互日志回退。"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import IO, Literal, Protocol, Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

ProgressStage = Literal[
    "run_started",
    "turn_started",
    "provider_started",
    "provider_streaming",
    "provider_finished",
    "tool_started",
    "context_window_reset",
    "restart",
    "verification",
    "turn_completed",
    "run_completed",
    "error",
]


@dataclass(frozen=True)
class ProgressUpdate:
    """一次与展示实现解耦的 benchmark 进度更新。"""

    stage: ProgressStage
    task_id: str
    variant: str
    repeat: int
    total_turns: int
    attempt: int = 1
    turn: int | None = None
    detail: str = ""


class BenchmarkProgressReporter(Protocol):
    """CLI 进度展示的最小协议。"""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def update(self, event: ProgressUpdate) -> None: ...

    def add_runs(self, count: int) -> None: ...


class RichBenchmarkProgress:
    """交互终端中的双层 Rich 进度条。"""

    def __init__(self, total_runs: int, console: Console) -> None:
        self._lock = threading.RLock()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}", markup=False),
            console=console,
            refresh_per_second=8,
        )
        self._overall: TaskID = self._progress.add_task(
            "overall",
            total=total_runs,
            status="waiting",
        )
        self._current: TaskID = self._progress.add_task(
            "current",
            total=1,
            status="waiting",
            visible=False,
        )

    def __enter__(self) -> Self:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def update(self, event: ProgressUpdate) -> None:
        with self._lock:
            label = _run_label(event)
            status = _status(event)
            if event.stage == "run_started":
                self._progress.update(
                    self._current,
                    description=label,
                    total=event.total_turns,
                    completed=0,
                    status=status,
                    visible=True,
                )
                self._progress.update(self._overall, status=label)
                return
            if event.stage == "turn_started" and event.turn is not None:
                self._progress.update(
                    self._current,
                    completed=max(0, event.turn - 1),
                    status=status,
                )
                return
            if event.stage == "turn_completed" and event.turn is not None:
                self._progress.update(
                    self._current,
                    completed=event.turn,
                    status=status,
                )
                return
            if event.stage == "run_completed":
                self._progress.update(
                    self._current,
                    completed=event.total_turns,
                    status=status,
                )
                self._progress.advance(self._overall)
                self._progress.update(self._overall, status=f"finished {label}")
                return
            self._progress.update(self._current, status=status)

    def add_runs(self, count: int) -> None:
        """在发生重试时扩展总运行数。"""
        if count <= 0:
            return
        with self._lock:
            task = self._progress.tasks[self._overall]
            self._progress.update(self._overall, total=(task.total or 0) + count)


class PlainBenchmarkProgress:
    """管道、CI 和重定向输出中的逐行进度日志。"""

    def __init__(self, total_runs: int, stream: IO[str] = sys.stderr) -> None:
        self._total_runs = total_runs
        self._completed_runs = 0
        self._stream = stream
        self._lock = threading.RLock()
        self._last_provider_activity_log = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def update(self, event: ProgressUpdate) -> None:
        with self._lock:
            if event.stage == "provider_started":
                self._last_provider_activity_log = 0.0
            elif event.stage == "provider_streaming":
                now = time.monotonic()
                if now - self._last_provider_activity_log < 30:
                    return
                self._last_provider_activity_log = now
            if event.stage == "run_completed":
                self._completed_runs += 1
            timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
            overall = f"run {self._completed_runs}/{self._total_runs}"
            turn = (
                f" turn {event.turn}/{event.total_turns}"
                if event.turn is not None
                else ""
            )
            print(
                f"[{timestamp}] {overall} {_run_label(event)}{turn}: {_status(event)}",
                file=self._stream,
                flush=True,
            )

    def add_runs(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._total_runs += count


class NullBenchmarkProgress:
    """显式关闭进度输出时使用的空实现。"""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def update(self, event: ProgressUpdate) -> None:
        return None

    def add_runs(self, count: int) -> None:
        return None


def create_progress_reporter(
    total_runs: int,
    *,
    enabled: bool = True,
    console: Console | None = None,
) -> BenchmarkProgressReporter:
    """按终端能力选择动态进度条或逐行日志。"""
    if not enabled:
        return NullBenchmarkProgress()
    active_console = console or Console(stderr=True)
    if active_console.is_terminal:
        return RichBenchmarkProgress(total_runs, active_console)
    return PlainBenchmarkProgress(total_runs, active_console.file)


def _run_label(event: ProgressUpdate) -> str:
    attempt = f" a{event.attempt}" if event.attempt > 1 else ""
    return f"{event.task_id} {event.variant} r{event.repeat}{attempt}"


def _status(event: ProgressUpdate) -> str:
    default = {
        "run_started": "preparing workspace",
        "turn_started": "starting turn",
        "provider_started": "waiting for first model event",
        "provider_streaming": "model request active",
        "provider_finished": "model response received",
        "tool_started": "running tool",
        "context_window_reset": "opening fresh context window",
        "restart": "rebuilding session",
        "verification": "running verification",
        "turn_completed": "turn completed",
        "run_completed": "run completed",
        "error": "failed",
    }[event.stage]
    detail = " ".join(event.detail.split())
    if not detail:
        return default
    if len(detail) > 96:
        detail = detail[:93] + "..."
    return f"{default}: {detail}"
