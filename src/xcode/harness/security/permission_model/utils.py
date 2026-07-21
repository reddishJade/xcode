from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .protocols import GrantStore
from .types import (
    Action,
    ApprovalCandidate,
    BLOCKED_PATH_PARTS,
    BoundaryContext,
    CREDENTIAL_PATH_PARTS,
    DirAccess,
    FingerprintLookupResult,
    GrantDecision,
    GrantRecord,
    GrantScope,
    PermissionAccess,
    Target,
    TargetFingerprint,
)

_COMMAND_GRANT_ARITY: dict[tuple[str, ...], int] = {
    ("git",): 2,
    ("npm", "run"): 3,
    ("pnpm", "run"): 3,
    ("yarn", "run"): 3,
    ("bun", "run"): 3,
    ("aws", "s3", "ls"): 3,
}


def command_grant_pattern(command: str) -> str:
    """返回 shell 命令持久授权实际匹配的命令模式。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if not tokens:
        return command
    lowered = tuple(token.lower() for token in tokens)
    arity = 1
    for prefix, prefix_arity in _COMMAND_GRANT_ARITY.items():
        if lowered[: len(prefix)] == prefix:
            arity = min(prefix_arity, len(tokens))
            break
    prefix = " ".join(tokens[:arity])
    return f"{prefix} *"


def _grant_target_pattern(target: Target) -> str:
    if target.kind == "command":
        return command_grant_pattern(target.value)
    return target.value


def create_grant_record(
    action: Action,
    target: Target,
    *,
    decision: GrantDecision,
    scope: GrantScope,
    grant_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> GrantRecord:
    return GrantRecord(
        capability=action.capability,
        operation=action.operation,
        target_kind=target.kind,
        target_pattern=_grant_target_pattern(target),
        access=target.access,
        decision=decision,
        scope=scope,
        grant_id=grant_id or uuid4().hex,
        metadata=metadata or {},
    )


def _compute_would_resolve(
    results: Sequence[FingerprintLookupResult],
) -> Literal["allow", "deny", "would_call_approval"]:
    for r in results:
        if r.grant is not None and r.grant.decision == "deny":
            return "deny"
    if all(r.grant is not None and r.grant.decision == "allow" for r in results):
        return "allow"
    return "would_call_approval"


def compute_shadow_approval_candidate(
    action: Action,
    *,
    session_grant_store: GrantStore | None = None,
    permanent_grant_store: GrantStore | None = None,
    boundary_context: BoundaryContext | None = None,
) -> ApprovalCandidate | None:
    if not action.targets:
        return None

    results: list[FingerprintLookupResult] = []
    for target in action.targets:
        fp = TargetFingerprint(
            capability=action.capability,
            operation=action.operation,
            target_kind=target.kind,
            target_pattern=target.value,
            access=target.access,
        )

        if session_grant_store is not None:
            grant = session_grant_store.lookup(
                action, target, boundary_context=boundary_context
            )
            if grant is not None:
                results.append(
                    FingerprintLookupResult(
                        fingerprint=fp, source="new_session", grant=grant
                    )
                )
                continue

        if permanent_grant_store is not None:
            grant = permanent_grant_store.lookup(
                action, target, boundary_context=boundary_context
            )
            if grant is not None:
                results.append(
                    FingerprintLookupResult(
                        fingerprint=fp, source="new_permanent", grant=grant
                    )
                )
                continue

        results.append(
            FingerprintLookupResult(fingerprint=fp, source="none", grant=None)
        )

    return ApprovalCandidate(
        would_resolve=_compute_would_resolve(results),
        fingerprints=tuple(results),
    )


def _is_external_path(path: str) -> bool:
    return _looks_absolute(path) or ".." in path.split("/")


def _looks_absolute(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("/") or (
        len(normalized) >= 3
        and normalized[1] == ":"
        and normalized[2] == "/"
        and normalized[0].isalpha()
    )


def _is_inside_path(candidate: Path, root: Path) -> bool:
    return candidate == root or candidate.is_relative_to(root)


def _access_satisfies(dir_access: DirAccess, target_access: PermissionAccess) -> bool:
    if dir_access == "read_write":
        return True
    if dir_access == "read":
        return target_access == "read"
    if dir_access == "write":
        return target_access in ("write",)


def _validate_symlinks_can_resolve(root: Path, relative_path: str) -> None:
    current = root
    for part in _relative_path_parts(relative_path):
        current = current / part
        if not current.is_symlink():
            continue
        current.stat()


def _relative_path_parts(relative_path: str) -> tuple[str, ...]:
    return tuple(part for part in relative_path.split("/") if part not in ("", "."))


def _is_git_path(path: str) -> bool:
    parts = tuple(part for part in path.split("/") if part)
    return ".git" in parts


def _is_sensitive_path(path: str, *, access: PermissionAccess = "read") -> bool:
    name = Path(path).name

    if name == ".env.example":
        return access == "write"

    if name == ".env" or name.startswith(".env."):
        return True

    parts = tuple(part for part in path.split("/") if part)
    return any(part in CREDENTIAL_PATH_PARTS for part in parts)


def _is_blocked_workspace_path(path: str) -> bool:
    parts = tuple(part for part in path.split("/") if part)
    if any(part in BLOCKED_PATH_PARTS for part in parts):
        return True
    return ".local" in parts and "chroma_db" in parts
