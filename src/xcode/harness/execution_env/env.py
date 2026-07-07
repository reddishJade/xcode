from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from .filesystem import FileSystem
from .result import ExecutionResult
from .shell import Shell


@runtime_checkable
class ExecutionEnv(Protocol):
    @property
    def fs(self) -> FileSystem: ...
    @property
    def shell(self) -> Shell: ...

    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30_000,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...
