from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
import json
from pathlib import Path
import re

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
    dynamic_decision: str
    policy_decision: str | None
    final_status: str
    approved: bool
    redacted_input: str
    redacted_output: str
    timestamp: str = ""
    turn_id: str = ""
    request_id: str = ""
    tool_call_id: str = ""
    approval_scope: str | None = None
    user_decision: str | None = None
    capability: str | None = None
    target_kind: str | None = None
    target_value: str | None = None
    matched_rule: str | None = None
    approval_source: str | None = None
    approval_grant_id: str | None = None

    def to_dict(self, timestamp: str | None = None) -> dict[str, str | bool | None]:
        created_at = self.timestamp or timestamp or datetime.now(UTC).isoformat()
        return {
            "session_id": self.session_id,
            "tool": self.tool,
            "dynamic_decision": self.dynamic_decision,
            "policy_decision": self.policy_decision,
            "final_status": self.final_status,
            "approved": self.approved,
            "redacted_input": self.redacted_input,
            "redacted_output": self.redacted_output,
            "timestamp": created_at,
            "turn_id": self.turn_id,
            "request_id": self.request_id,
            "tool_call_id": self.tool_call_id,
            "approval_scope": self.approval_scope,
            "user_decision": self.user_decision,
            "capability": self.capability,
            "target_kind": self.target_kind,
            "target_value": self.target_value,
            "matched_rule": self.matched_rule,
            "approval_source": self.approval_source,
            "approval_grant_id": self.approval_grant_id,
        }


class JsonlAuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: AuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def redact_text(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_redact_match, text)
    return text


def _redact_match(match: re.Match[str]) -> str:
    if len(match.groups()) >= 2:
        return f"{match.group(1)}=[REDACTED]"
    return "[REDACTED]"
