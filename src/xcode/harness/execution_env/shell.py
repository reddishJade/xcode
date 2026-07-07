from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .result import ExecutionResult


class Shell(Protocol):
    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30_000,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...
