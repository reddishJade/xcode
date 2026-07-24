from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PermissionAccess = Literal["read", "write", "execute", "network", "delete"]
DirAccess = Literal["read", "write", "read_write"]
GrantDecision = Literal["allow", "deny"]
GrantScope = Literal["once", "session", "permanent"]
PermissionDecisionV2 = Literal["allow", "ask", "deny"]
TargetKind = Literal["path", "command", "domain", "mcp", "subagent", "skill"]
ProvenanceKind = Literal["structured_arg", "shell_literal"]

CREDENTIAL_PATH_PARTS: frozenset[str] = frozenset(
    {
        ".aws",
        ".azure",
        ".docker",
        ".gnupg",
        ".kube",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".ssh",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
BLOCKED_PATH_PARTS: frozenset[str] = frozenset({".venv", "__pycache__"})
UnresolvedReason = Literal[
    "variable_expansion",
    "glob",
    "command_substitution",
    "wrapper_command",
    "eval_like",
    "parse_error",
    "unsupported_shell",
    "dangerous_command",
]

type GrantRecordData = dict[str, object]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(description="工具名或通配符，如 'bash', 'write_file', '*'")
    effect: PermissionDecisionV2 = Field(description="决策: allow / ask / deny")

    command: str | None = Field(
        default=None,
        description="shell 主命令，如 'git', 'rm', 'docker'",
    )
    subcommand: str | None = Field(
        default=None,
        description="精确子命令，如 'push'",
    )
    subcommand_in: set[str] | None = Field(
        default=None,
        description="匹配任一子命令，如 {'status', 'diff', 'log'}",
    )
    flags_any: set[str] | None = Field(
        default=None,
        description="含任一 flag 即匹配，如 {'--force', '-f'}",
    )
    flags_all: set[str] | None = Field(
        default=None,
        description="含全部 flag 才匹配",
    )

    resource_pattern: str | None = Field(
        default=None,
        description=(
            "通配符路径/资源模式。非 shell 工具使用此字段匹配 target.value；"
            "shell 工具中此项作为额外路径约束。"
        ),
    )


class ExternalDirectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    access: DirAccess = "read"

    @model_validator(mode="after")
    def _resolve_path(self) -> ExternalDirectory:
        self.path = self.path.expanduser().resolve(strict=False)
        return self


class SensitivePathOverride(BaseModel):
    """对单个环境配置文件的显式访问例外。"""

    model_config = ConfigDict(extra="forbid")
    path: Path
    access: DirAccess = "read"

    @model_validator(mode="after")
    def _resolve_path(self) -> SensitivePathOverride:
        self.path = self.path.expanduser().resolve(strict=False)
        return self


class StaticPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    decision: PermissionDecisionV2
    target: str | None = None
    target_type: Literal["path", "command", "mcp", "subagent", "skill", None] = None
    input_contains: str | None = None
    input_prefix: str | None = None
    input_regex: str | None = None


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: TargetKind
    value: str
    access: PermissionAccess
    provenance: ProvenanceKind = "structured_arg"


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    capability: str
    operation: str
    targets: tuple[Target, ...]
    input: Mapping[str, object]  # noqa: F821
    unresolved_effects: tuple[UnresolvedEffect, ...] = ()


class UnresolvedEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: UnresolvedReason
    fragment: str


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: PermissionDecisionV2
    source: str
    reason: str
    target_pattern: str | None = None
    operation: str | None = None
    access: PermissionAccess | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)  # noqa: F821


class BoundaryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_root: Path
    external_directories: tuple[ExternalDirectory, ...] = ()
    sensitive_path_overrides: tuple[SensitivePathOverride, ...] = ()


class ApprovalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["allow", "deny"]
    scope: Literal["once", "session", "permanent"]
    grant_id: str | None = None


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: PermissionDecisionV2
    source: str
    reason: str
    winning_constraint: Constraint | None
    constraints: tuple[Constraint, ...]
    approval: ApprovalResult | None = None
    grant_id: str | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)  # noqa: F821


class GrantRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    operation: str
    target_kind: TargetKind
    target_pattern: str
    access: PermissionAccess
    decision: GrantDecision
    scope: GrantScope
    grant_id: str
    metadata: Mapping[str, object] = Field(default_factory=dict)  # noqa: F821


class TargetFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    operation: str
    target_kind: TargetKind
    target_pattern: str
    access: PermissionAccess


class FingerprintLookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint: TargetFingerprint
    source: Literal["new_session", "new_permanent", "none"]
    grant: GrantRecord | None


class ApprovalCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    would_resolve: Literal["allow", "deny", "would_call_approval"]
    fingerprints: tuple[FingerprintLookupResult, ...]
