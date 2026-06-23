from __future__ import annotations

import threading


class CancellationToken(threading.Event):
    """供 Agent 循环和工具协作退出的轻量取消标记。"""

    def __init__(self) -> None:
        super().__init__()
        self._reason = "interrupted by user"

    def cancel(self, reason: str = "interrupted by user") -> None:
        self._reason = reason
        self.set()

    def reset(self) -> None:
        self._reason = "interrupted by user"
        self.clear()

    def is_cancelled(self) -> bool:
        return self.is_set()

    @property
    def reason(self) -> str:
        return self._reason
