"""四轴权限模型的数据结构、动作提取和约束解析。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
from typing import Literal, Protocol
from uuid import uuid4

PermissionAccess = Literal["read", "write", "execute", "network"]
GrantDecision = Literal["allow", "deny"]
GrantScope = Literal["once", "session", "permanent"]
PermissionDecisionV2 = Literal["allow", "ask", "deny"]
TargetKind = Literal["path", "command", "domain", "mcp", "subagent"]
type GrantRecordData = dict[str, object]


@dataclass(frozen=True)
class Target:
    """动作作用的具体对象。"""

    kind: TargetKind
    value: str
    access: PermissionAccess


@dataclass(frozen=True)
class Action:
    """一次工具调用归一化后的权限判断输入。"""

    tool: str
    capability: str
    operation: str
    targets: tuple[Target, ...]
    input: Mapping[str, object]


@dataclass(frozen=True)
class Constraint:
    """单个策略对 action 给出的约束。"""

    decision: PermissionDecisionV2
    source: str
    reason: str
    non_bypassable: bool = False
    target_pattern: str | None = None
    operation: str | None = None
    access: PermissionAccess | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryContext:
    """结构化边界策略所需的文件系统上下文。"""

    project_root: Path


@dataclass(frozen=True)
class ApprovalResult:
    """用户或 reviewer 对 ask verdict 的授权结果。"""

    decision: Literal["allow", "deny"]
    scope: Literal["once", "session", "permanent"]
    grant_id: str | None = None


@dataclass(frozen=True)
class Verdict:
    """resolver 合并所有 constraints 后的最终结论。"""

    decision: PermissionDecisionV2
    source: str
    reason: str
    winning_constraint: Constraint | None
    constraints: tuple[Constraint, ...]
    approval: ApprovalResult | None = None
    grant_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GrantRecord:
    """结构化授权记录。"""

    capability: str
    operation: str
    target_kind: TargetKind
    target_pattern: str
    access: PermissionAccess
    decision: GrantDecision
    scope: GrantScope
    grant_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetFingerprint:
    """授权请求中单个目标的指纹。"""

    capability: str
    operation: str
    target_kind: TargetKind
    target_pattern: str
    access: PermissionAccess


@dataclass(frozen=True)
class FingerprintLookupResult:
    """单个目标指纹的查找结果。"""

    fingerprint: TargetFingerprint
    source: Literal["new_session", "new_permanent", "none"]
    grant: GrantRecord | None


@dataclass(frozen=True)
class ApprovalCandidate:
    """shadow 模型对 resolver 返回 ask 后 candidate 路径的预测结果。

    不会调用真实 approval_callback，不写入任何 grant store。
    """

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
    """会话级内存授权存储。"""

    def __init__(self, records: Iterable[GrantRecord] = ()) -> None:
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


class StaticPermissionPolicy(Protocol):
    """legacy 静态权限策略的最小接口。"""

    def decide(self, tool_name: str, action_input: str) -> PermissionDecisionV2 | None:
        """按 legacy 静态规则优先级返回权限决策。"""
        ...


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
                results.append(FingerprintLookupResult(fp, "new_session", grant))
                continue

        if permanent_grant_store is not None:
            grant = permanent_grant_store.lookup(
                action, target, boundary_context=boundary_context
            )
            if grant is not None:
                results.append(FingerprintLookupResult(fp, "new_permanent", grant))
                continue

        results.append(FingerprintLookupResult(fp, "none", None))

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
    capability = value.get("capability")
    operation = value.get("operation")
    target_kind = value.get("target_kind")
    target_pattern = value.get("target_pattern")
    access = value.get("access")
    decision = value.get("decision")
    scope = value.get("scope")
    grant_id = value.get("grant_id")
    metadata = value.get("metadata")
    if not isinstance(capability, str):
        return None
    if not isinstance(operation, str):
        return None
    if not isinstance(target_kind, str):
        return None
    if not isinstance(target_pattern, str):
        return None
    if not isinstance(access, str):
        return None
    if not isinstance(decision, str):
        return None
    if not isinstance(scope, str):
        return None
    if not isinstance(grant_id, str):
        return None
    if target_kind not in ("path", "command", "domain", "mcp", "subagent"):
        return None
    if access not in ("read", "write", "execute", "network"):
        return None
    if decision not in ("allow", "deny"):
        return None
    if scope not in ("once", "session", "permanent"):
        return None
    return GrantRecord(
        capability=capability,
        operation=operation,
        target_kind=target_kind,
        target_pattern=target_pattern,
        access=access,
        decision=decision,
        scope=scope,
        grant_id=grant_id,
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def _grant_record_to_data(record: GrantRecord) -> GrantRecordData:
    data: GrantRecordData = {
        "capability": record.capability,
        "operation": record.operation,
        "target_kind": record.target_kind,
        "target_pattern": record.target_pattern,
        "access": record.access,
        "decision": record.decision,
        "scope": record.scope,
        "grant_id": record.grant_id,
    }
    if record.metadata:
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
    """把 legacy 静态规则和 allowlist 模式转换为约束。"""

    def __init__(
        self,
        policy: StaticPermissionPolicy | None = None,
        *,
        allowlist_mode: bool = False,
        action_input: str | None = None,
    ) -> None:
        self._policy = policy
        self._allowlist_mode = allowlist_mode
        self._action_input = action_input

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        action_input = self._serialized_action_input(action)
        decision = self._policy_decision(action, action_input)
        if decision is not None:
            return self._constraints_for_action(
                action,
                decision,
                f"static permission rule returned {decision} for {action.tool}",
            )

        if self._allowlist_mode:
            return self._constraints_for_action(
                action,
                "ask",
                f"allowlist mode has no allow rule for {action.tool}",
            )

        return ()

    def _policy_decision(
        self, action: Action, action_input: str
    ) -> PermissionDecisionV2 | None:
        if self._policy is None:
            return None
        return self._policy.decide(action.tool, action_input)

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
        path = target.value
        if _is_external_path(path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"path escapes workspace boundary: {path}",
                target_pattern=path,
                operation=action.operation,
                access=target.access,
            )

        if self._context is not None:
            try:
                path = self._resolve_workspace_path(target)
            except _BoundaryEscapeError:
                return Constraint(
                    decision="deny",
                    source="boundary",
                    reason=f"path escapes workspace boundary: {path}",
                    target_pattern=path,
                    operation=action.operation,
                    access=target.access,
                )
            except _BoundaryResolutionError as exc:
                return Constraint(
                    decision="deny",
                    source="boundary",
                    reason=f"path cannot be resolved safely: {path}: {exc}",
                    non_bypassable=target.access in ("write", "execute"),
                    target_pattern=path,
                    operation=action.operation,
                    access=target.access,
                )

        if _is_git_path(path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"git metadata path is blocked: {path}",
                non_bypassable=target.access == "write",
                target_pattern=path,
                operation=action.operation,
                access=target.access,
            )

        if _is_sensitive_path(path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"sensitive path is blocked: {path}",
                target_pattern=path,
                operation=action.operation,
                access=target.access,
            )

        if _is_blocked_workspace_path(path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"workspace blocked path is denied: {path}",
                target_pattern=path,
                operation=action.operation,
                access=target.access,
            )

        return Constraint(
            decision="allow",
            source="boundary",
            reason=f"workspace-internal path is allowed: {path}",
            target_pattern=path,
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


class SafetyBackstopPolicyEvaluator:
    """对 shell 命令进行三桶分类的不可绕过安全策略。

    按 section 10.2 定义的三桶模式对 shell 命令逐段评估：
    - Bucket A: non-bypassable deny
    - Bucket B: ask
    - Bucket C: allow
    - 未匹配: ask（安全默认）

    Compound 命令按 section 10.4 拆分后逐段独立评估，
    整体 verdict 取最严格约束。
    """

    BUCKET_A_CREDENTIAL_SUBSTRINGS: tuple[str, ...] = (
        ".ssh/",
        ".ssh",
        ".gnupg/",
        ".gnupg",
        ".aws/",
        ".aws",
        ".gcloud/",
        ".gcloud",
        ".config/git/credentials",
        ".netrc",
    )

    BUCKET_A_DEVICE_PATTERNS: tuple[str, ...] = (
        "mkfs.",
        "> /dev/sda",
        "> /dev/sdb",
        "> /dev/nvme",
    )

    BUCKET_B_ASK_COMMANDS: set[str] = {
        "rm",
        "mv",
        "curl",
        "wget",
        "kill",
        "pkill",
        "killall",
        "nohup",
        "disown",
        "tmux",
        "screen",
        "sudo",
    }

    BUCKET_C_ALLOW_COMMANDS: set[str] = {
        "ls",
        "dir",
        "find",
        "rg",
        "grep",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "pwd",
        "which",
        "type",
        "realpath",
        "echo",
        "printf",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "diff",
        "cmp",
        "comm",
        "du",
        "df",
        "ruff",
        "pyright",
        "mypy",
        "flake8",
        "eslint",
        "pytest",
        "unittest",
        "make",
        "uname",
        "env",
        "export",
        "cd",
        "npx",
        "prettier",
    }

    BUCKET_B_GIT_PREFIXES: tuple[str, ...] = (
        "git reset --hard",
        "git clean -f",
        "git push --force",
        "git push -f",
        "git config --global",
    )

    BUCKET_C_GIT_PREFIXES: tuple[str, ...] = (
        "git status",
        "git diff",
        "git log",
        "git show",
        "git branch",
        "git stash list",
        "git stash show",
        "git remote -v",
        "git tag",
        "git config --list",
        "git config --get",
    )

    BUCKET_A_GIT_CREDENTIAL_PATTERNS: tuple[str, ...] = (
        "git credential",
        "git config --global credential",
    )

    BUCKET_A_PACKAGE_PATTERNS: tuple[str, ...] = (
        "apt install",
        "apt-get install",
        "yum install",
        "dnf install",
        "brew install",
    )

    BUCKET_A_SERVICE_PATTERNS: tuple[str, ...] = (
        "systemctl start",
        "systemctl stop",
        "systemctl enable",
        "systemctl disable",
        "systemctl restart",
        "systemctl reload",
        "service ",
        "initctl ",
    )

    BUCKET_B_DOCKER_PREFIXES: tuple[str, ...] = (
        "docker build",
        "docker run",
        "docker push",
        "docker commit",
    )

    BUCKET_C_CHECK_PREFIXES: tuple[str, ...] = (
        "ruff format --check",
        "prettier --check",
        "ruff check",
        "tsc --noEmit",
        "python --version",
        "node --version",
        "go version",
        "rustc --version",
        "cargo --version",
        "pip list",
        "pip show",
        "pip freeze",
        "npm list",
        "npm outdated",
        "npx --help",
        "cargo test",
        "cargo build",
        "cargo check",
        "go test",
        "go build",
        "go vet",
        "go fmt",
    )

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        if action.capability != "shell":
            return ()
        command = self._get_command(action)
        if not command:
            return ()
        segments = _split_compound_command(command)
        constraints: list[Constraint] = []
        for segment in segments:
            constraint = self._evaluate_segment(segment)
            if constraint is not None:
                constraints.append(constraint)
        return tuple(constraints)

    def _get_command(self, action: Action) -> str:
        for target in action.targets:
            if target.kind == "command":
                return target.value
        return ""

    def _evaluate_segment(self, segment: str) -> Constraint | None:
        stripped = segment.strip()
        if not stripped:
            return None

        bucket_a = self._check_bucket_a(stripped)
        if bucket_a is not None:
            return bucket_a

        bucket_b = self._check_bucket_b(stripped)
        if bucket_b is not None:
            return bucket_b

        bucket_c = self._check_bucket_c(stripped)
        if bucket_c is not None:
            return bucket_c
        return Constraint(
            decision="ask",
            source="safety_backstop",
            reason=f"未识别的命令，安全默认 ask: {stripped[:200]}",
        )

    def _check_bucket_a(self, command: str) -> Constraint | None:
        """Bucket A: 必须 non-bypassable deny 的模式。"""
        normalized = command.strip().lower()

        if _is_root_recursive_deletion(command):
            return self._deny_constraint("根目录递归删除操作", non_bypassable=True)

        if _is_system_path_recursive_mutation(command):
            return self._deny_constraint("系统关键路径递归破坏", non_bypassable=True)

        if _is_dd_device_write(command):
            return self._deny_constraint("dd 直接设备写入", non_bypassable=True)

        for pattern in self.BUCKET_A_DEVICE_PATTERNS:
            if pattern in normalized:
                return self._deny_constraint(
                    f"裸设备写入: {pattern}", non_bypassable=True
                )

        if _is_root_recursive_permission_change(command):
            return self._deny_constraint("根目录递归权限修改", non_bypassable=True)

        for pattern in self.BUCKET_A_CREDENTIAL_SUBSTRINGS:
            if pattern in command:
                return self._deny_constraint(
                    f"凭据路径访问: {pattern}", non_bypassable=True
                )

        for pattern in self.BUCKET_A_GIT_CREDENTIAL_PATTERNS:
            if pattern in normalized:
                return self._deny_constraint(
                    f"git 凭据 helper 调用: {pattern}", non_bypassable=True
                )

        for pattern in self.BUCKET_A_PACKAGE_PATTERNS:
            if normalized.startswith(pattern) or f" {pattern}" in normalized:
                return self._deny_constraint(
                    f"系统包管理器安装: {pattern}", non_bypassable=True
                )

        for pattern in self.BUCKET_A_SERVICE_PATTERNS:
            if normalized.startswith(pattern) or f" {pattern}" in normalized:
                return self._deny_constraint(
                    f"系统服务控制: {pattern}", non_bypassable=True
                )

        return None

    def _check_bucket_b(self, command: str) -> Constraint | None:
        """Bucket B: 必须 ask 的模式。"""
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None

        first = tokens[0].lower()

        if first in self.BUCKET_B_ASK_COMMANDS:
            return self._ask_constraint(f"高风险命令: {first}")

        if first == "git" and len(tokens) > 1:
            git_cmd = " ".join(tokens).lower()
            for prefix in self.BUCKET_B_GIT_PREFIXES:
                if git_cmd.startswith(prefix):
                    return self._ask_constraint(f"强制 git 操作: {prefix}")

        for opname in ("chmod", "chown"):
            if first == opname and _has_recursive_flag(tokens[1:]):
                return self._ask_constraint(f"递归权限修改: {opname}")

        if first == "docker":
            docker_cmd = " ".join(tokens).lower()
            for prefix in self.BUCKET_B_DOCKER_PREFIXES:
                if docker_cmd.startswith(prefix):
                    return self._ask_constraint(f"Docker 变异操作: {prefix}")

        return None

    def _check_bucket_c(self, command: str) -> Constraint | None:
        """Bucket C: 已知安全可 allow 的模式。"""
        if _segment_has_redirect_or_substitution(command):
            return None

        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None

        first = tokens[0].lower()

        if first == "git" and len(tokens) > 1:
            git_cmd = " ".join(tokens).lower()
            for prefix in self.BUCKET_C_GIT_PREFIXES:
                if git_cmd.startswith(prefix):
                    return self._allow_constraint(f"git 只读操作: {prefix}")

        check_cmd = command.strip().lower()
        for prefix in self.BUCKET_C_CHECK_PREFIXES:
            if check_cmd.startswith(prefix):
                return self._allow_constraint(f"已知安全命令: {prefix}")

        if first in self.BUCKET_C_ALLOW_COMMANDS:
            return self._allow_constraint(f"已知安全命令: {first}")

        if first == "command" and len(tokens) > 1 and tokens[1] == "-v":
            return self._allow_constraint("已知安全命令: command -v")

        if first == "git" and len(tokens) == 1:
            return self._allow_constraint("已知安全命令: git")

        return None

    def _deny_constraint(
        self, reason: str, *, non_bypassable: bool = False
    ) -> Constraint:
        return Constraint(
            decision="deny",
            source="safety_backstop",
            reason=reason,
            non_bypassable=non_bypassable,
        )

    def _ask_constraint(self, reason: str) -> Constraint:
        return Constraint(
            decision="ask",
            source="safety_backstop",
            reason=reason,
        )

    def _allow_constraint(self, reason: str) -> Constraint:
        return Constraint(
            decision="allow",
            source="safety_backstop",
            reason=reason,
        )


def _split_compound_command(command: str) -> list[str]:
    """按顶层操作符拆分 shell 复合命令，忽略引号和 $()/反引号内操作符。

    操作符: && || ; | \\n
    $() 和反引号不拆分，视为单段 opaque。
    """
    # 预处理：将 \\n（反斜杠换行续行）替换为空格，避免续行后的操作符被拆出独立段
    command = _normalize_backslash_continuation(command)
    segments: list[str] = []
    current: list[str] = []
    i = 0
    n = len(command)
    in_single = False
    in_double = False
    in_backtick = False
    paren_depth = 0

    while i < n:
        ch = command[i]

        if in_single:
            current.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            current.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue

        if in_backtick:
            current.append(ch)
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if paren_depth > 0:
            current.append(ch)
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            i += 1
            continue

        if ch == "'":
            in_single = True
            current.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double = True
            current.append(ch)
            i += 1
            continue

        if ch == "`":
            in_backtick = True
            current.append(ch)
            i += 1
            continue

        if ch == "$" and i + 1 < n and command[i + 1] == "(":
            paren_depth = 1
            current.append(ch)
            current.append("(")
            i += 2
            continue

        sep_len = 0
        if ch == ";":
            sep_len = 1
        elif ch == "|" and i + 1 < n and command[i + 1] == "|":
            sep_len = 2
        elif ch == "&" and i + 1 < n and command[i + 1] == "&":
            sep_len = 2
        elif ch == "|":
            sep_len = 1
        elif ch == "\n":
            sep_len = 1

        if sep_len:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            i += sep_len
            continue

        current.append(ch)
        i += 1

    segment = "".join(current).strip()
    if segment:
        segments.append(segment)

    return segments


def _normalize_backslash_continuation(command: str) -> str:
    """将反斜杠换行续行（\\\\n）替换为空格。"""
    return command.replace("\\\n", " ")


def _is_dd_device_write(command: str) -> bool:
    """检查 dd 命令是否写入 /dev/ 设备。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] != "dd":
        return False
    return any(token.startswith("of=/dev/") for token in tokens[1:])


def _is_root_recursive_deletion(command: str) -> bool:
    """检查是否为 rm -rf / 类根目录递归删除（双 pass，不依赖 flag 顺序）。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] != "rm":
        return False
    has_recursive = any(
        _is_short_flag_with_r(t) for t in tokens[1:] if t.startswith("-") and t != "--"
    )
    if not has_recursive:
        return False
    for token in tokens[1:]:
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        cleaned = token.rstrip("/")
        if cleaned in {"", "/", "/*"}:
            return True
    return False


def _is_system_path_recursive_mutation(command: str) -> bool:
    """检查是否为系统关键路径递归破坏。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if len(tokens) < 3 or tokens[0] not in {"rm", "mv", "chmod", "chown"}:
        return False
    if not _has_recursive_flag(tokens[1:]):
        return False
    return any(_is_protected_system_target(token) for token in tokens[1:])


def _is_protected_system_target(token: str) -> bool:
    protected = {
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    }
    cleaned = token.rstrip("/")
    if cleaned in {"", "/", "/*"}:
        return True
    return cleaned in protected


def _is_root_recursive_permission_change(command: str) -> bool:
    """检查是否为 chmod -R 777 / 等根目录递归权限修改。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] not in {"chmod", "chown"}:
        return False
    if not _has_recursive_flag(tokens[1:]):
        return False
    for token in tokens[1:]:
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        cleaned = token.rstrip("/")
        if cleaned in {"", "/", "/*"}:
            return True
    return False


def _is_short_flag_with_r(token: str) -> bool:
    """检查 token 是否为含 r/R 的短 flag 组合（-rf, -Rf, -r 等）。

    要求：
    - 单破折号开头（-- 开头的不算）；
    - 去掉前导 - 后全部为字母，长度 <= 5，且至少含一个 r/R。
    长度限制排除 -version、-format 等单破折号长 flag。
    """
    if not token.startswith("-") or token.startswith("--"):
        return False
    stripped = token.lstrip("-")
    if not stripped or not stripped.isalpha() or len(stripped) > 5:
        return False
    return "r" in stripped.lower()


def _has_recursive_flag(tokens: list[str]) -> bool:
    """检查 token 列表中是否有递归短 flag（-r 或 -R）。"""
    for token in tokens:
        if token == "--":
            return False
        if _is_short_flag_with_r(token):
            return True
    return False


def _segment_has_redirect_or_substitution(segment: str) -> bool:
    """检查命令段是否包含重定向或命令替换。

    含重定向或命令替换的段不能进入 Bucket C。
    """
    if "$(" in segment:
        return True
    if "`" in segment:
        return True
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return True
    for token in tokens:
        if token in (">", ">>", ">&", "<>", "<", "<<", "<<<", "<("):
            return True
        if token == "tee":
            return True
    return False


class _BoundaryEscapeError(ValueError):
    """路径解析后离开工作区边界。"""


class _BoundaryResolutionError(ValueError):
    """路径无法安全解析。"""


def evaluate_policy_constraints(
    action: Action,
    *,
    execution_decision: PermissionDecisionV2 | None = None,
    static_policy: StaticPermissionPolicy | None = None,
    allowlist_mode: bool = False,
    action_input: str | None = None,
    boundary_context: BoundaryContext | None = None,
    safety_backstop_enabled: bool = False,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
) -> tuple[Constraint, ...]:
    """运行已接入的 shadow policy evaluators 和 hook constraint providers。

    Hook constraint providers 在所有内置 evaluator 之后执行，
    产生的 constraint 进入同一池由 resolver 按 standard priority 处理。
    """
    evaluators: list[PolicyEvaluator] = [
        ModePolicyEvaluator(execution_decision),
        StaticPolicyEvaluator(
            static_policy,
            allowlist_mode=allowlist_mode,
            action_input=action_input,
        ),
        StructuredBoundaryPolicyEvaluator(boundary_context),
    ]
    if safety_backstop_enabled:
        evaluators.append(SafetyBackstopPolicyEvaluator())
    evaluators.extend(hook_constraint_providers)
    constraints: list[Constraint] = []
    for evaluator in evaluators:
        constraints.extend(evaluator.evaluate(action))
    return tuple(constraints)


class ActionExtractor:
    """把工具调用保守归一化为 Action。"""

    def extract(self, tool_name: str, tool_input: Mapping[str, object]) -> Action:
        """根据工具名和结构化输入提取 action。"""
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
        return Action(
            tool=tool_name,
            capability="unknown",
            operation=tool_name,
            targets=(),
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
            targets = (Target("path", _normalize_path_text(raw_path), access),)
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
            Target("path", _normalize_path_text(path), "write")
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
            targets = (Target("command", command.strip(), "execute"),)
        return Action(
            tool=tool_name,
            capability="shell",
            operation="run_command",
            targets=targets,
            input=tool_input,
        )

    def _shell_action(self, tool_name: str, tool_input: Mapping[str, object]) -> Action:
        targets = tuple(
            Target("command", command.strip(), "execute")
            for command in self._shell_commands(tool_input)
            if command.strip()
        )
        return Action(
            tool=tool_name,
            capability="shell",
            operation="run_command",
            targets=targets,
            input=tool_input,
        )

    def _patch_paths(self, tool_input: Mapping[str, object]) -> tuple[str, ...]:
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
    return path == ".git" or path.startswith(".git/")


def _is_sensitive_path(path: str) -> bool:
    parts = tuple(part for part in path.split("/") if part)
    return any(_is_sensitive_part(part) for part in parts)


def _is_sensitive_part(part: str) -> bool:
    if part == ".env" or part.startswith(".env."):
        return True
    return part in StructuredBoundaryPolicyEvaluator.CREDENTIAL_PATH_PARTS


def _is_blocked_workspace_path(path: str) -> bool:
    parts = tuple(part for part in path.split("/") if part)
    if any(
        part in StructuredBoundaryPolicyEvaluator.BLOCKED_PATH_PARTS for part in parts
    ):
        return True
    return ".local" in parts and "chroma_db" in parts
