"""审计与 provider request 记录。"""

from __future__ import annotations

from collections.abc import Callable

from ...agent.config import AfterToolCallContext
from ..observability import AuditRecord, redact_text
from ..skills import ToolSpec


def emit_audit(
    audit_logger: Callable[[AuditRecord], None] | None,
    session_id: str,
    ctx: AfterToolCallContext,
    action_input: str,
    result_text: str,
    tool_map: dict[str, ToolSpec],
) -> None:
    if audit_logger is None:
        return
    tool_call = ctx.tool_call
    spec = tool_map.get(tool_call.name)
    audit_logger(
        AuditRecord(
            session_id=session_id,
            tool=tool_call.name,
            static_risk=(spec.risk if spec else None) or "low",
            dynamic_decision="allow",
            policy_decision=None,
            final_status="error" if ctx.is_error else "ok",
            approved=True,
            redacted_input=redact_text(action_input),
            redacted_output=redact_text(result_text),
        )
    )
