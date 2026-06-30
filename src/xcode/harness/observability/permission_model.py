"""四轴权限模型的数据结构、动作提取和约束解析。

本模块职责（请勿盲目拆分）：
- 核心权限数据模型：Action, Target, Constraint, Verdict, GrantRecord, etc.
- GrantStore 协议及实现：InMemoryGrantStore, FileGrantStore
- PermissionResolver（约束优先级解析）
- ModePolicyEvaluator, StaticPolicyEvaluator, StructuredBoundaryPolicyEvaluator
- evaluate_policy_constraints（编排所有 evaluator）
- ActionExtractor 及路径辅助函数

SafetyBackstopPolicyEvaluator 及其 shell 命令分类已拆至 _safety_backstop.py。
_safety_backstop 从本模块导入 Action/Constraint 等共享模型类型；
本模块在 evaluate_policy_constraints 内局部导入 SafetyBackstopPolicyEvaluator
以避免 import-time 循环。其余分组（grant store / resolver / evaluators / extractor）
紧密关联，拆分需要谨慎评估依赖边界。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

PermissionAccess = Literal["read", "write", "execute", "network"]
DirAccess = Literal["read", "write", "read_write"]
GrantDecision = Literal["allow", "deny"]
GrantScope = Literal["once", "session", "permanent"]
PermissionDecisionV2 = Literal["allow", "ask", "deny"]
TargetKind = Literal["path", "command", "domain", "mcp", "subagent", "skill"]
type GrantRecordData = dict[str, object]

logger = logging.getLogger(__name__)


class ExternalDirectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    access: DirAccess = "read"

    @model_validator(mode="after")
    def _resolve_path(self) -> ExternalDirectory:
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
    """动作作用的具体对象。"""

    model_config = ConfigDict(extra="forbid")
    kind: TargetKind
    value: str
    access: PermissionAccess


class Action(BaseModel):
    """一次工具调用归一化后的权限判断输入。"""

    model_config = ConfigDict(extra="forbid")
    tool: str
    capability: str
    operation: str
    targets: tuple[Target, ...]
    input: Mapping[str, object]


class Constraint(BaseModel):
    """单个策略对 action 给出的约束。"""

    model_config = ConfigDict(extra="forbid")
    decision: PermissionDecisionV2
    source: str
    reason: str
    non_bypassable: bool = False
    target_pattern: str | None = None
    operation: str | None = None
    access: PermissionAccess | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)


class BoundaryContext(BaseModel):
    """结构化边界策略所需的文件系统上下文。"""

    model_config = ConfigDict(extra="forbid")
    project_root: Path
    external_directories: tuple[ExternalDirectory, ...] = ()


class ApprovalResult(BaseModel):
    """用户或 reviewer 对 ask verdict 的授权结果。"""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["allow", "deny"]
    scope: Literal["once", "session", "permanent"]
    grant_id: str | None = None


class Verdict(BaseModel):
    """resolver 合并所有 constraints 后的最终结论。"""

    model_config = ConfigDict(extra="forbid")
    decision: PermissionDecisionV2
    source: str
    reason: str
    winning_constraint: Constraint | None
    constraints: tuple[Constraint, ...]
    approval: ApprovalResult | None = None
    grant_id: str | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)


class GrantRecord(BaseModel):
    """结构化授权记录。"""

    model_config = ConfigDict(extra="forbid")

    capability: str
    operation: str
    target_kind: TargetKind
    target_pattern: str
    access: PermissionAccess
    decision: GrantDecision
    scope: GrantScope
    grant_id: str
    metadata: Mapping[str, object] = Field(default_factory=dict)


class TargetFingerprint(BaseModel):
    """授权请求中单个目标的指纹。"""

    model_config = ConfigDict(extra="forbid")

    capability: str
    operation: str
    target_kind: TargetKind
    target_pattern: str
    access: PermissionAccess


class FingerprintLookupResult(BaseModel):
    """单个目标指纹的查找结果。"""

    model_config = ConfigDict(extra="forbid")

    fingerprint: TargetFingerprint
    source: Literal["new_session", "new_permanent", "none"]
    grant: GrantRecord | None


class ApprovalCandidate(BaseModel):
    """shadow 模型对 resolver 返回 ask 后 candidate 路径的预测结果。

    不会调用真实 approval_callback，不写入任何 grant store。
    """

    model_config = ConfigDict(extra="forbid")

    would_resolve: Literal["allow", "deny", "would_call_approval"]
    fingerprints: tuple[FingerprintLookupResult, ...]


class GrantStore(Protocol):
    """结构化授权记录存储接口。"""

    def add(self, record: GrantRecord) -> GrantRecord:
        """写入或替换一条授权记录。"""
        ...

    def records(self) -> tuple[GrantRecord, ...]:
        """返回当前全部授权记录。"""
        ...

    def lookup(
        self,
        action: Action,
        target: Target,
        *,
        boundary_context: BoundaryContext | None = None,
    ) -> GrantRecord | None:
        """查找与 action target 匹配的授权记录。"""
        ...


class InMemoryGrantStore:
    """会话级内存授权存储。

    通过 session_id 标识所属会话。不同会话的实例不应共享。
    """

    def __init__(
        self,
        records: Iterable[GrantRecord] = (),
        *,
        session_id: str = "",
    ) -> None:
        self._session_id = session_id
        self._records = tuple(records)

    def add(self, record: GrantRecord) -> GrantRecord:
        """按 grant_id 替换旧记录并写入新记录。"""
        self._records = tuple(
            existing
            for existing in self._records
            if existing.grant_id != record.grant_id
        ) + (record,)
        return record

    def records(self) -> tuple[GrantRecord, ...]:
        """返回当前全部授权记录。"""
        return self._records

    def lookup(
        self,
        action: Action,
        target: Target,
        *,
        boundary_context: BoundaryContext | None = None,
    ) -> GrantRecord | None:
        """按 deny > allow 查找匹配授权。"""
        return _lookup_grant_record(
            self._records,
            action,
            target,
            boundary_context=boundary_context,
        )

    def clear(self) -> None:
        """清空会话授权记录。"""
        self._records = ()


class SessionGrantStoreManager:
    """管理 session_id 到 InMemoryGrantStore 的映射。

    同一 logical_session_id 在同一进程中复用同一 store。
    进程重启后所有 session grants 丢失。
    不管理 FileGrantStore（永久授权与会话无关）。
    """

    def __init__(self) -> None:
        self._stores: dict[str, InMemoryGrantStore] = {}

    def get_for_session(self, session_id: str) -> InMemoryGrantStore:
        if session_id not in self._stores:
            self._stores[session_id] = InMemoryGrantStore(session_id=session_id)
        return self._stores[session_id]


class FileGrantStore:
    """文件持久化的结构化授权记录，默认写入 .local/approval_grants.json。"""

    DEFAULT_RELATIVE_PATH = Path(".local") / "approval_grants.json"

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_project_root(cls, project_root: Path) -> FileGrantStore:
        """按项目根目录创建默认持久化授权存储。"""
        return cls(project_root / cls.DEFAULT_RELATIVE_PATH)

    def add(self, record: GrantRecord) -> GrantRecord:
        """按 grant_id 替换旧记录并写入新记录。"""
        updated = tuple(
            existing
            for existing in self.records()
            if existing.grant_id != record.grant_id
        ) + (record,)
        self._write(updated)
        return record

    def records(self) -> tuple[GrantRecord, ...]:
        """读取并过滤无效授权记录。"""
        if not self.path.exists():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ()
        if not isinstance(raw, list):
            return ()
        return tuple(
            record
            for item in raw
            if (record := _grant_record_from_data(item)) is not None
        )

    def lookup(
        self,
        action: Action,
        target: Target,
        *,
        boundary_context: BoundaryContext | None = None,
    ) -> GrantRecord | None:
        """按 deny > allow 查找匹配授权。"""
        return _lookup_grant_record(
            self.records(),
            action,
            target,
            boundary_context=boundary_context,
        )

    def _write(self, records: tuple[GrantRecord, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [_grant_record_to_data(record) for record in records]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


class PermissionResolver:
    """按固定优先级把 constraints 解析为最终 verdict。"""

    DEFAULT_SOURCE = "resolver"
    DEFAULT_REASON = "no constraints produced; default allow"

    def resolve(self, constraints: tuple[Constraint, ...]) -> Verdict:
        """按 non-bypassable deny > deny > ask > allow 解析约束。"""
        if not constraints:
            return Verdict(
                decision="allow",
                source=self.DEFAULT_SOURCE,
                reason=self.DEFAULT_REASON,
                winning_constraint=None,
                constraints=constraints,
            )

        winner = self._winning_constraint(constraints)
        return Verdict(
            decision=winner.decision,
            source=winner.source,
            reason=winner.reason,
            winning_constraint=winner,
            constraints=constraints,
            metadata=winner.metadata,
        )

    def _winning_constraint(self, constraints: tuple[Constraint, ...]) -> Constraint:
        non_bypassable_denies = tuple(
            c for c in constraints if c.decision == "deny" and c.non_bypassable
        )
        if non_bypassable_denies:
            return non_bypassable_denies[0]

        explicit_denies = tuple(c for c in constraints if c.decision == "deny")
        if explicit_denies:
            return explicit_denies[0]

        asks = tuple(c for c in constraints if c.decision == "ask")
        if asks:
            return asks[0]

        allows = tuple(c for c in constraints if c.decision == "allow")
        if allows:
            return allows[0]

        raise ValueError("constraint decision must be allow, ask, or deny")


class PolicyEvaluator(Protocol):
    """根据 action 生成权限约束。"""

    def evaluate(self, action: Action) -> tuple[Constraint, ...]: ...


def create_grant_record(
    action: Action,
    target: Target,
    *,
    decision: GrantDecision,
    scope: GrantScope,
    grant_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> GrantRecord:
    """从 action target 创建结构化授权记录。"""
    return GrantRecord(
        capability=action.capability,
        operation=action.operation,
        target_kind=target.kind,
        target_pattern=target.value,
        access=target.access,
        decision=decision,
        scope=scope,
        grant_id=grant_id or uuid4().hex,
        metadata=metadata or {},
    )


def _compute_would_resolve(
    results: Sequence[FingerprintLookupResult],
) -> Literal["allow", "deny", "would_call_approval"]:
    """根据逐 target 查找结果计算整体 candidate 决策。"""
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
    """构造 shadow approval candidate：predict engine-level grant/callback 结果。

    只读/observational，不调用 approval_callback，不写入任何 grant store。
    对命令类型的 target 同样适用。
    """
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


def _lookup_grant_record(
    records: tuple[GrantRecord, ...],
    action: Action,
    target: Target,
    *,
    boundary_context: BoundaryContext | None = None,
) -> GrantRecord | None:
    matching = tuple(
        record
        for record in records
        if _grant_matches_target(
            record,
            action,
            target,
            boundary_context=boundary_context,
        )
    )
    if not matching:
        return None
    return _highest_priority_grant(matching)


def _highest_priority_grant(records: Sequence[GrantRecord]) -> GrantRecord:
    for record in records:
        if record.decision == "deny":
            return record
    return records[0]


def _grant_matches_target(
    record: GrantRecord,
    action: Action,
    target: Target,
    *,
    boundary_context: BoundaryContext | None = None,
) -> bool:
    if record.capability != action.capability:
        return False
    if record.operation != action.operation:
        return False
    if record.target_kind != target.kind:
        return False
    if record.access != target.access:
        return False
    if target.kind != "path":
        return record.target_pattern == target.value
    return _path_pattern_matches(
        record.target_pattern,
        target.value,
        boundary_context=boundary_context,
    )


def _path_pattern_matches(
    target_pattern: str,
    candidate: str,
    *,
    boundary_context: BoundaryContext | None = None,
) -> bool:
    pattern = _normalize_target_path(target_pattern, boundary_context=boundary_context)
    normalized_candidate = _normalize_target_path(
        candidate,
        boundary_context=boundary_context,
    )
    if pattern == normalized_candidate:
        return True
    return normalized_candidate.startswith(f"{pattern}/")


def _normalize_target_path(
    path: str,
    *,
    boundary_context: BoundaryContext | None = None,
) -> str:
    normalized = _normalize_path_text(path)
    if boundary_context is None or _is_external_path(normalized):
        return normalized

    root = boundary_context.project_root
    try:
        resolved_root = root.resolve(strict=False)
        candidate = (resolved_root / normalized).resolve(strict=False)
    except (OSError, RuntimeError):
        return normalized
    if not _is_inside_path(candidate, resolved_root):
        return normalized
    return candidate.relative_to(resolved_root).as_posix() or "."


def _grant_record_from_data(value: object) -> GrantRecord | None:
    if not isinstance(value, dict):
        return None
    try:
        return GrantRecord.model_validate(value)
    except ValidationError:
        return None


def _grant_record_to_data(record: GrantRecord) -> GrantRecordData:
    data: GrantRecordData = record.model_dump(exclude_none=True)
    if not record.metadata:
        data.pop("metadata", None)
    else:
        data["metadata"] = dict(record.metadata)
    return data


class ModePolicyEvaluator:
    """把当前执行模式判定转换为约束。"""

    def __init__(self, decision: PermissionDecisionV2 | None) -> None:
        self._decision: PermissionDecisionV2 | None = decision

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        if self._decision is None:
            return ()
        return (
            Constraint(
                decision=self._decision,
                source="mode",
                reason=f"mode policy returned {self._decision} for {action.tool}",
                operation=action.operation,
            ),
        )


class StaticPolicyEvaluator:
    """把静态权限规则通过 last-match-wins 转换为约束。

    Rules 按声明顺序遍历，最后一个匹配的 rule 的 decision 生效。
    如果无规则匹配且设置了 global_default，发出一个 global_default 约束。
    如果无规则匹配且无 global_default，不发出约束。
    """

    def __init__(
        self,
        rules: tuple[StaticPermission, ...] = (),
        *,
        global_default: PermissionDecisionV2 | None = None,
        action_input: str | None = None,
    ) -> None:
        self._rules = rules
        self._global_default = global_default
        self._action_input = action_input

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        action_input = self._serialized_action_input(action)
        decision = self._match_rules(action, action_input)
        if decision is not None:
            return self._constraints_for_action(
                action,
                decision,
                f"static permission rule returned {decision} for {action.tool}",
            )
        gd = self._global_default
        if gd is not None:
            return self._constraints_for_action(
                action,
                cast("PermissionDecisionV2", gd),
                f"no static rule matched; global_default={gd}",
            )
        return ()

    def _match_rules(
        self, action: Action, action_input: str
    ) -> PermissionDecisionV2 | None:
        last: PermissionDecisionV2 | None = None
        for rule in self._rules:
            if rule.tool != action.tool and rule.tool != "*":
                continue
            if (
                rule.input_contains is not None
                and rule.input_contains not in action_input
            ):
                continue
            if rule.input_prefix is not None and not action_input.startswith(
                rule.input_prefix
            ):
                continue
            if rule.input_regex is not None and not re.search(
                rule.input_regex, action_input
            ):
                continue
            if rule.target is not None:
                if not any(target.value == rule.target for target in action.targets):
                    continue
            if rule.target_type is not None:
                if not any(
                    target.kind == rule.target_type for target in action.targets
                ):
                    continue
            last = rule.decision
        return last

    def _serialized_action_input(self, action: Action) -> str:
        if self._action_input is not None:
            return self._action_input
        return json.dumps(action.input, ensure_ascii=False, sort_keys=True)

    def _constraints_for_action(
        self, action: Action, decision: PermissionDecisionV2, reason: str
    ) -> tuple[Constraint, ...]:
        if not action.targets:
            return (
                Constraint(
                    decision=decision,
                    source="rule",
                    reason=reason,
                    operation=action.operation,
                ),
            )

        return tuple(
            Constraint(
                decision=decision,
                source="rule",
                reason=reason,
                target_pattern=target.value,
                operation=action.operation,
                access=target.access,
            )
            for target in action.targets
        )


class StructuredBoundaryPolicyEvaluator:
    """结构化文件工具的路径边界策略。"""

    STRUCTURED_TOOLS = frozenset(
        {
            "read_file",
            "write_file",
            "edit_file",
            "apply_patch",
            "grep_search",
            "glob_files",
            "find_files",
            "list_dir",
        }
    )
    CREDENTIAL_PATH_PARTS = frozenset(
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
    BLOCKED_PATH_PARTS = frozenset({".venv", "__pycache__"})

    def __init__(self, context: BoundaryContext | None = None) -> None:
        self._context = context

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        if action.tool not in self.STRUCTURED_TOOLS:
            return ()

        constraints: list[Constraint] = []
        for target in action.targets:
            if target.kind != "path":
                continue
            constraints.append(self._path_constraint(action, target))
        return tuple(constraints)

    def _path_constraint(self, action: Action, target: Target) -> Constraint:
        path_str = target.value

        if self._context is None:
            if _is_external_path(path_str):
                return Constraint(
                    decision="deny",
                    source="boundary",
                    reason=f"path escapes workspace boundary: {path_str}",
                    target_pattern=path_str,
                    operation=action.operation,
                    access=target.access,
                )
            return self._check_restrictions(path_str, path_str, action, target)

        # Three-way classification with BoundaryContext
        try:
            resolved = self._resolve_workspace_path(target)
            # resolved is relative to workspace root
            return self._check_restrictions(resolved, path_str, action, target)
        except _BoundaryEscapeError:
            candidate = self._try_external_directory(target, action)
            if candidate is not None:
                return candidate
            logger.warning(
                "path resolved outside workspace boundary: %s",
                path_str,
            )
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"path outside all approved roots: {path_str}",
                target_pattern=path_str,
                operation=action.operation,
                access=target.access,
            )
        except _BoundaryResolutionError as exc:
            candidate = self._try_external_directory(target, action)
            if candidate is not None:
                return candidate
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"path cannot be resolved safely: {path_str}: {exc}",
                non_bypassable=target.access in ("write", "execute"),
                target_pattern=path_str,
                operation=action.operation,
                access=target.access,
            )

    def _try_external_directory(
        self, target: Target, action: Action
    ) -> Constraint | None:
        """If target is inside an approved external_directory with sufficient
        access, run security checks and return a constraint.

        Returns None if no external_directory matches.
        """
        assert self._context is not None
        raw = target.value
        resolved_root = self._context.project_root.resolve(strict=False)

        try:
            if _looks_absolute(raw):
                candidate = Path(raw).resolve(strict=False)
            else:
                candidate = (resolved_root / raw).resolve(strict=False)
        except (OSError, RuntimeError):
            return None

        for ext in self._context.external_directories:
            if not _is_inside_path(candidate, ext.path):
                continue
            if not _access_satisfies(ext.access, target.access):
                continue
            check = candidate.as_posix()
            return self._check_restrictions(check, raw, action, target)
        return None

    def _check_restrictions(
        self,
        check_path: str,
        original_path: str,
        action: Action,
        target: Target,
    ) -> Constraint:
        """Run git, sensitive, and blocked-path checks on a path that has
        already been classified as inside an approved root."""
        if _is_git_path(check_path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"git metadata path is blocked: {original_path}",
                non_bypassable=target.access == "write",
                target_pattern=check_path,
                operation=action.operation,
                access=target.access,
            )

        if _is_sensitive_path(check_path, access=target.access):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"sensitive path is blocked: {original_path}",
                target_pattern=check_path,
                operation=action.operation,
                access=target.access,
            )

        if _is_blocked_workspace_path(check_path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"workspace blocked path is denied: {original_path}",
                target_pattern=check_path,
                operation=action.operation,
                access=target.access,
            )

        return Constraint(
            decision="allow",
            source="boundary",
            reason=f"path is allowed: {original_path}",
            target_pattern=check_path,
            operation=action.operation,
            access=target.access,
        )

    def _resolve_workspace_path(self, target: Target) -> str:
        assert self._context is not None
        root = self._context.project_root
        try:
            resolved_root = root.resolve(strict=False)
            source = resolved_root / target.value
            _validate_symlinks_can_resolve(resolved_root, target.value)
            candidate = source.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise _BoundaryResolutionError(str(exc)) from exc

        if not _is_inside_path(candidate, resolved_root):
            raise _BoundaryEscapeError(target.value)

        return candidate.relative_to(resolved_root).as_posix() or "."


class _BoundaryEscapeError(ValueError):
    """路径解析后离开工作区边界。"""


class _BoundaryResolutionError(ValueError):
    """路径无法安全解析。"""


def evaluate_policy_constraints(
    action: Action,
    *,
    execution_decision: PermissionDecisionV2 | None = None,
    static_policy: Any = None,
    action_input: str | None = None,
    boundary_context: BoundaryContext | None = None,
    safety_backstop_enabled: bool = False,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
) -> tuple[Constraint, ...]:
    """运行已接入的 shadow policy evaluators 和 hook constraint providers。

    static_policy: PermissionPolicy | None — 携带 rules 和 global_default。

    Hook constraint providers 在所有内置 evaluator 之后执行，
    产生的 constraint 进入同一池由 resolver 按 standard priority 处理。
    """
    rules: tuple[StaticPermission, ...] = ()
    global_default: PermissionDecisionV2 | None = None
    if static_policy is not None:
        rules = static_policy.rules
        gd = static_policy.global_default
        if gd is not None:
            global_default = cast("PermissionDecisionV2", gd)

    evaluators: list[Any] = [
        ModePolicyEvaluator(execution_decision),
        StaticPolicyEvaluator(
            rules,
            global_default=global_default,
            action_input=action_input,
        ),
        StructuredBoundaryPolicyEvaluator(boundary_context),
    ]
    if safety_backstop_enabled:
        from ._safety_backstop import SafetyBackstopPolicyEvaluator

        evaluators.append(SafetyBackstopPolicyEvaluator())
    evaluators.extend(hook_constraint_providers)
    constraints: list[Constraint] = []
    for evaluator in evaluators:
        constraints.extend(evaluator.evaluate(action))
    return tuple(constraints)


class ActionExtractor:
    """把工具调用保守归一化为 Action。"""

    def extract(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        action_profile: tuple[str, str] | None = None,
    ) -> Action:
        """根据工具名和结构化输入提取 action。

        当提供 action_profile 时，使用其 capability 覆盖硬编码映射，
        但保留已有的 target 提取逻辑。
        """
        action = self._extract_inner(tool_name, tool_input)
        if action_profile is not None:
            capability_name, _ = action_profile
            action = Action(
                tool=action.tool,
                capability=capability_name,
                operation=action.operation,
                targets=action.targets,
                input=action.input,
            )
        return action

    def _extract_inner(
        self, tool_name: str, tool_input: Mapping[str, object]
    ) -> Action:
        """原始工具名→Action 映射，不含 action_profile 覆盖。"""
        if tool_name == "read_file":
            return self._path_action(tool_name, tool_input, "read", "read_file", "read")
        if tool_name == "write_file":
            return self._path_action(
                tool_name, tool_input, "write", "write_file", "write"
            )
        if tool_name == "edit_file":
            return self._path_action(
                tool_name, tool_input, "edit", "edit_file", "write"
            )
        if tool_name == "apply_patch":
            return self._apply_patch_action(tool_name, tool_input)
        if tool_name == "bash":
            return self._bash_action(tool_name, tool_input)
        if tool_name == "shell":
            return self._shell_action(tool_name, tool_input)
        if tool_name == "delete_file":
            return self._path_action(
                tool_name, tool_input, "write", "delete_file", "write"
            )
        if tool_name in ("grep_search", "glob_files", "find_files", "list_dir"):
            return self._path_action(
                tool_name, tool_input, "read", tool_name, "read"
            )
        if tool_name == "load_skill":
            return self._load_skill_action(tool_name, tool_input)
        if tool_name.startswith("mcp__"):
            return Action(
                tool=tool_name,
                capability="mcp",
                operation=tool_name,
                targets=(Target(kind="mcp", value=tool_name, access="execute"),),
                input=tool_input,
            )
        return Action(
            tool=tool_name,
            capability="unknown",
            operation=tool_name,
            targets=(),
            input=tool_input,
        )

    def _load_skill_action(
        self, tool_name: str, tool_input: Mapping[str, object]
    ) -> Action:
        name = tool_input.get("name")
        targets: tuple[Target, ...] = ()
        if isinstance(name, str) and name.strip():
            targets = (Target(kind="skill", value=name.strip(), access="read"),)
        return Action(
            tool=tool_name,
            capability="skill",
            operation="load_skill",
            targets=targets,
            input=tool_input,
        )

    def _path_action(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        capability: str,
        operation: str,
        access: PermissionAccess,
    ) -> Action:
        raw_path = tool_input.get("path")
        targets: tuple[Target, ...] = ()
        if isinstance(raw_path, str) and raw_path.strip():
            targets = (
                Target(
                    kind="path", value=_normalize_path_text(raw_path), access=access
                ),
            )
        return Action(
            tool=tool_name,
            capability=capability,
            operation=operation,
            targets=targets,
            input=tool_input,
        )

    def _apply_patch_action(
        self, tool_name: str, tool_input: Mapping[str, object]
    ) -> Action:
        targets = tuple(
            Target(kind="path", value=_normalize_path_text(path), access="write")
            for path in self._patch_paths(tool_input)
        )
        return Action(
            tool=tool_name,
            capability="patch",
            operation="apply_patch",
            targets=targets,
            input=tool_input,
        )

    def _bash_action(self, tool_name: str, tool_input: Mapping[str, object]) -> Action:
        command = tool_input.get("command")
        targets: tuple[Target, ...] = ()
        if isinstance(command, str) and command.strip():
            normalized_command = command.strip()
            targets = (
                Target(kind="command", value=normalized_command, access="execute"),
                *self._shell_path_targets(normalized_command),
            )
        return Action(
            tool=tool_name,
            capability="shell",
            operation="run_command",
            targets=targets,
            input=tool_input,
        )

    def _shell_action(self, tool_name: str, tool_input: Mapping[str, object]) -> Action:
        targets: list[Target] = []
        for command in self._shell_commands(tool_input):
            normalized_command = command.strip()
            if not normalized_command:
                continue
            targets.append(
                Target(kind="command", value=normalized_command, access="execute")
            )
            targets.extend(self._shell_path_targets(normalized_command))
        return Action(
            tool=tool_name,
            capability="shell",
            operation="run_command",
            targets=tuple(targets),
            input=tool_input,
        )

    def _shell_path_targets(self, command: str) -> tuple[Target, ...]:
        """从已知文件系统命令中保守提取路径参数。"""
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return ()
        if not tokens:
            return ()

        command_name = Path(tokens[0].strip("\"'")).name.lower()
        path_arguments = _filesystem_command_path_arguments(command_name, tokens[1:])
        access: PermissionAccess = (
            "read" if command_name in _READ_FILESYSTEM_COMMANDS else "write"
        )
        return tuple(
            Target(kind="path", value=_normalize_path_text(path), access=access)
            for path in path_arguments
        )

    def _patch_paths(self, tool_input: Mapping[str, object]) -> tuple[str, ...]:
        from xcode.coding_agent.tools.apply_patch import extract_patch_paths

        extracted = extract_patch_paths(dict(tool_input))
        if extracted:
            return extracted

        raw_path = tool_input.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            return (raw_path,)

        raw_paths = tool_input.get("paths")
        if isinstance(raw_paths, tuple | list):
            return tuple(path for path in raw_paths if isinstance(path, str))

        return ()

    def _shell_commands(self, tool_input: Mapping[str, object]) -> tuple[str, ...]:
        raw_commands = tool_input.get("commands")
        if isinstance(raw_commands, tuple | list):
            return tuple(
                command for command in raw_commands if isinstance(command, str)
            )

        raw_command = tool_input.get("command")
        if isinstance(raw_command, str):
            return (raw_command,)

        return ()


def _normalize_path_text(raw_path: str) -> str:
    path = raw_path.strip()
    if _looks_absolute(path):
        return path.replace("\\", "/")
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        parts.append(part)
    return "/".join(parts) or "."


_READ_FILESYSTEM_COMMANDS = frozenset(
    {
        "cat",
        "dir",
        "get-childitem",
        "get-content",
        "head",
        "less",
        "ls",
        "more",
        "realpath",
        "tail",
    }
)
_WRITE_FILESYSTEM_COMMANDS = frozenset(
    {
        "copy-item",
        "cp",
        "del",
        "move-item",
        "mv",
        "remove-item",
        "rm",
        "set-content",
    }
)


def _filesystem_command_path_arguments(
    command_name: str,
    arguments: Sequence[str],
) -> tuple[str, ...]:
    """返回已知文件系统命令的位置路径参数。"""
    if command_name not in _READ_FILESYSTEM_COMMANDS | _WRITE_FILESYSTEM_COMMANDS:
        return ()

    paths: list[str] = []
    for argument in arguments:
        cleaned = argument.strip("\"'")
        if not cleaned or cleaned.startswith("-"):
            continue
        if cleaned in {"&&", "||", ";", "|"}:
            break
        paths.append(cleaned)
    return tuple(paths)


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
    return any(
        part in StructuredBoundaryPolicyEvaluator.CREDENTIAL_PATH_PARTS
        for part in parts
    )


def _is_blocked_workspace_path(path: str) -> bool:
    parts = tuple(part for part in path.split("/") if part)
    if any(
        part in StructuredBoundaryPolicyEvaluator.BLOCKED_PATH_PARTS for part in parts
    ):
        return True
    return ".local" in parts and "chroma_db" in parts
