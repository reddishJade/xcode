from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
import json
from pathlib import Path
import re
from typing import Any

"""工具执行审计与敏感信息脱敏。"""


SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{8,})"
    ),
)


@dataclass(frozen=True)
class AuditRecord:
    session_id: str
    tool: str
    static_risk: str
    dynamic_decision: str
    policy_decision: str | None
    final_status: str
    approved: bool
    redacted_input: str
    redacted_output: str
    timestamp: str = ""
    approval_scope: str | None = None
    user_decision: str | None = None


class JsonlAuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: AuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        if not payload["timestamp"]:
            payload["timestamp"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def redact_text(value: Any) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_redact_match, text)
    return text


def _redact_match(match: re.Match[str]) -> str:
    if len(match.groups()) >= 2:
        return f"{match.group(1)}=[REDACTED]"
    return "[REDACTED]"
