"""审计、权限和钩子能力。

trace 依赖 RAG 类型，使用时从 `observability.tracing` 直接导入，避免聚合
入口和 RAG pipeline 形成循环依赖。
"""

from .audit import AuditRecord, JsonlAuditLogger, redact_text
from .hooks import (
    BeforeAgentStartEvent,
    BeforeProviderRequestEvent,
    CompactEvent,
    ErrorEvent,
    HarnessEvent,
    HookManager,
    HookRecord,
    PostToolEvent,
    PreToolEvent,
)
from .permissions import (
    HITLResult,
    PermissionApprovalCallback,
    PermissionCheckResult,
    PermissionDecision,
    PermissionPolicy,
    PermissionRiskEvaluator,
    PermissionRule,
    PermissionToolSpec,
    PersistentPermissionStore,
    SessionPermissionPolicy,
    SettingsSandboxPermissionPolicy,
    CompositePermissionPolicy,
    check_tool_permission,
)

__all__ = [
    "AuditRecord",
    "BeforeAgentStartEvent",
    "BeforeProviderRequestEvent",
    "CompactEvent",
    "ErrorEvent",
    "HarnessEvent",
    "HITLResult",
    "HookManager",
    "HookRecord",
    "JsonlAuditLogger",
    "PermissionApprovalCallback",
    "PermissionCheckResult",
    "PermissionDecision",
    "PermissionPolicy",
    "PermissionRiskEvaluator",
    "PermissionRule",
    "PermissionToolSpec",
    "PersistentPermissionStore",
    "SessionPermissionPolicy",
    "SettingsSandboxPermissionPolicy",
    "CompositePermissionPolicy",
    "PostToolEvent",
    "PreToolEvent",
    "check_tool_permission",
    "redact_text",
]
