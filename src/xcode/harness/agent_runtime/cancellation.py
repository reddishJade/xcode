from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class CancellationState:
    reason: str = "interrupted by user"


class CancellationToken:
    """供 Agent 循环和工具协作退出的轻量取消标记。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = CancellationState()

    def cancel(self, reason: str = "interrupted by user") -> None:
        self._reason = CancellationState(reason)
        self._event.set()

    def reset(self) -> None:
        self._reason = CancellationState()
        self._event.clear()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> threading.Event:
        return self._event

    @property
    def reason(self) -> str:
        return self._reason.reason
