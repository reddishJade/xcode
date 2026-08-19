"""审计与 provider request 记录。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...agent.config import AfterToolCallContext
from ...agent.types import ToolCallContent
from ..observability import AuditLogger, AuditRecord, HookCorrelationFields, redact_text

if TYPE_CHECKING:
    from ..security import PermissionEngineResult


def emit_audit(
    audit_logger: AuditLogger | None,
    session_id: str,
    ctx: AfterToolCallContext,
    action_input: str,
    result_text: str,
    perm_result: PermissionEngineResult | None = None,
    correlation: HookCorrelationFields | None = None,
) -> None:
    if audit_logger is None:
        return

    audit_logger(
        build_audit_record(
            session_id=session_id,
            tool_call=ctx.tool_call,
            action_input=action_input,
            result_text=result_text,
            final_status="error" if ctx.is_error else "ok",
            perm_result=perm_result,
            correlation=correlation,
        )
    )


def build_audit_record(
    *,
    session_id: str,
    tool_call: ToolCallContent,
    action_input: str,
    result_text: str,
    final_status: str,
    perm_result: PermissionEngineResult | None,
    correlation: HookCorrelationFields | None = None,
) -> AuditRecord:
    """构建统一的工具审计记录。"""
    metadata = (perm_result.metadata or {}) if perm_result is not None else {}
    approval_result = perm_result.approval_result if perm_result is not None else None
    action = perm_result.action if perm_result is not None else None
    target = action.targets[0] if action is not None and action.targets else None
    fields = correlation or {}
    return AuditRecord(
        session_id=session_id,
        tool=tool_call.name,
        dynamic_decision=perm_result.decision if perm_result is not None else "allow",
        policy_decision=(perm_result.matched_rule if perm_result is not None else None),
        final_status=final_status,
        approved=not perm_result.blocked if perm_result is not None else True,
        redacted_input=redact_text(action_input),
        redacted_output=redact_text(result_text),
        timestamp=fields.get("timestamp", ""),
        turn_id=fields.get("turn_id", ""),
        request_id=fields.get("request_id", ""),
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
        approval_decision=(
            str(metadata.get("approval_decision"))
            if metadata.get("approval_decision")
            else None
        ),
        capability=action.capability if action is not None else None,
        target_kind=str(target.kind) if target is not None else None,
        target_value=str(target.value) if target is not None else None,
        matched_rule=perm_result.matched_rule if perm_result is not None else None,
        approval_source=perm_result.source if perm_result is not None else None,
        approval_reviewer=(
            str(metadata.get("approval_reviewer"))
            if metadata.get("approval_reviewer")
            else None
        ),
        approval_review_status=(
            str(metadata.get("approval_review_status"))
            if metadata.get("approval_review_status")
            else None
        ),
        approval_rationale=(
            str(metadata.get("approval_rationale"))
            if metadata.get("approval_rationale")
            else None
        ),
        approval_risk=(
            str(metadata.get("approval_risk"))
            if metadata.get("approval_risk")
            else None
        ),
        approval_authorization=(
            str(metadata.get("approval_authorization"))
            if metadata.get("approval_authorization")
            else None
        ),
        approval_grant_id=(
            approval_result.grant_id if approval_result is not None else None
        ),
    )
