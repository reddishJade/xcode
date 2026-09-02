"""审计、关联与钩子能力。

权限相关类型请从 `xcode.harness.security` 导入。
"""

from .audit import AuditLogger, AuditRecord, JsonlAuditLogger, redact_text
from .correlation import (
    EventCorrelation,
    HookCorrelationFields,
    RuntimeCorrelation,
    hook_correlation_fields,
)
from .external_hooks import (
    ExternalHookDiagnostic,
    ExternalHookExecution,
    ExternalHookFailure,
    ExternalHookRunner,
)
from .hooks import (
    BeforeAgentStartEvent,
    BeforeProviderRequestEvent,
    ContextWindowResetHookEvent,
    ErrorEvent,
    HarnessEvent,
    HookManager,
    HookRecord,
    PostToolEvent,
    PreToolEvent,
    SignalHookManager,
)

__all__ = [
    "AuditLogger",
    "AuditRecord",
    "BeforeAgentStartEvent",
    "BeforeProviderRequestEvent",
    "ContextWindowResetHookEvent",
    "ErrorEvent",
    "EventCorrelation",
    "ExternalHookDiagnostic",
    "ExternalHookExecution",
    "ExternalHookFailure",
    "ExternalHookRunner",
    "HarnessEvent",
    "HookCorrelationFields",
    "HookManager",
    "HookRecord",
    "JsonlAuditLogger",
    "PostToolEvent",
    "PreToolEvent",
    "RuntimeCorrelation",
    "SignalHookManager",
    "hook_correlation_fields",
    "redact_text",
]
