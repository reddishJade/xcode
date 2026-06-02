from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    status: str = "ok"
    elapsed_ms: float | None = None
