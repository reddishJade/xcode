from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from xcode.harness.agent_runtime import StructuredAgentEvent


class TraceRecorder:
    """将 Agent 事件流落盘为 JSONL trace。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.count = 0

    def record(self, event: StructuredAgentEvent) -> None:
        self.count += 1
        payload = {
            "index": self.count,
            "type": event.type,
            "step": event.step,
            "data": _jsonable(event.data),
        }
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()

    def record_error(self, exc: BaseException) -> None:
        self.count += 1
        payload = {
            "index": self.count,
            "type": "runtime_error",
            "step": 0,
            "data": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)
