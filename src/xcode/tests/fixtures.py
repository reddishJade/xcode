from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.providers.faux import FauxProvider
from xcode.harness.execution_env import ExecutionResult


class MockExecutionEnv:
    """测试桩实现，记录调用并返回预设结果。"""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, int]] = []
        self._results: list[ExecutionResult] = []
        self._result_index = 0

    def enqueue(self, result: ExecutionResult) -> None:
        self._results.append(result)

    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[str], None] | None = None,  # noqa: ARG002
        env: dict[str, str] | None = None,  # noqa: ARG002
    ) -> ExecutionResult:
        self.calls.append((argv, cwd, timeout))
        if self._result_index < len(self._results):
            result = self._results[self._result_index]
            self._result_index += 1
            return result
        return ExecutionResult(
            stdout="", stderr="", returncode=0, timed_out=False, cancelled=False
        )


class FakeProvider(FauxProvider):
    """Lightweight alias for FauxProvider.

    Supports ProviderEvent lists, list-of-lists, or callable factories.
    Delegates to the consolidated FauxProvider implementation.
    """

    def __init__(
        self,
        events: list[ProviderEvent]
        | list[list[ProviderEvent]]
        | Callable[[list[Any], list[Any]], list[ProviderEvent]],
    ) -> None:
        super().__init__(response_spec=events)
