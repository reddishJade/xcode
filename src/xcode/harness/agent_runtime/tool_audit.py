"""审计与 provider request 记录。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ...agent.config import AfterToolCallContext
from ..observability import AuditRecord, HookCorrelationFields, redact_text
from ..skills import ToolSpec

if TYPE_CHECKING:
    from ..observability import PermissionEngineResult


def emit_audit(
    audit_logger: Callable[[AuditRecord], None] | None,
    session_id: str,
    ctx: AfterToolCallContext,
    action_input: str,
    result_text: str,
    tool_map: dict[str, ToolSpec],
    perm_result: PermissionEngineResult | None = None,
    correlation: HookCorrelationFields | None = None,
) -> None:
    if audit_logger is None:
        return
    tool_call = ctx.tool_call
    metadata = (perm_result.metadata or {}) if perm_result is not None else {}
    approval_result = perm_result.approval_result if perm_result is not None else None

    audit_logger(
        AuditRecord(
            session_id=session_id,
            tool=tool_call.name,
            dynamic_decision=(
                perm_result.decision if perm_result is not None else "allow"
            ),
            policy_decision=(
                perm_result.matched_rule if perm_result is not None else None
            ),
            final_status="error" if ctx.is_error else "ok",
            approved=(not perm_result.blocked if perm_result is not None else True),
            redacted_input=redact_text(action_input),
            redacted_output=redact_text(result_text),
            timestamp=(correlation or {}).get("timestamp", ""),
            turn_id=(correlation or {}).get("turn_id", ""),
            request_id=(correlation or {}).get("request_id", ""),
            tool_call_id=tool_call.id,
            approval_scope=(
                approval_result.scope
                if approval_result is not None and approval_result.scope
                else (
                    str(metadata.get("approval_scope"))
                    if metadata.get("approval_scope")
                    else None
                )
            ),
            user_decision=(
                str(metadata.get("user_decision"))
                if metadata.get("user_decision")
                else None
            ),
            capability=(
                perm_result.action.capability
                if perm_result is not None and perm_result.action is not None
                else None
            ),
            target_kind=(
                str(perm_result.action.targets[0].kind)
                if perm_result is not None
                and perm_result.action is not None
                and perm_result.action.targets
                else None
            ),
            target_value=(
                str(perm_result.action.targets[0].value)
                if perm_result is not None
                and perm_result.action is not None
                and perm_result.action.targets
                else None
            ),
            matched_rule=(
                perm_result.matched_rule if perm_result is not None else None
            ),
            approval_source=(perm_result.source if perm_result is not None else None),
            approval_grant_id=(
                approval_result.grant_id if approval_result is not None else None
            ),
        )
    )
