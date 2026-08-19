"""工具执行的 allow/deny/ask 权限策略与审批模型。

权限架构：静态策略、执行模式、已有授权与当前 mode 的 reviewer。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Literal

from xcode.agent.types import ApprovalRequest, ApprovalScope

from ..session import JsonValue
from .approval import (
    ApprovalPolicy,
    ApprovalsReviewer,
    HITLDecision,
    HITLScope,
    PermissionApprovalCallback,
    ReviewAuthorization,
    ReviewRisk,
    ReviewStatus,
)
from .permission_model import (
    Action,
    ActionExtractor,
    ApprovalCandidate,
    ApprovalResult,
    BoundaryContext,
    Constraint,
    ExternalDirectory,
    GrantDecision as _GrantDecision,
    GrantRecord,
    GrantScope as _GrantScope,
    GrantStore,
    PolicyEvaluator,
    PermissionResolver,
    PathExtractor,
    Rule,
    SensitivePathOverride,
    StaticPermission,
    Verdict,
    compute_approval_candidate,
    create_grant_record,
    evaluate_policy_constraints,
)
from .rule_matcher import first_match as rule_first_match, merge_rulesets as rule_merge


PermissionDecision = Literal["allow", "deny", "ask"]
ShellUnresolvedPolicy = Literal["allow", "deny", "ask"]
type PermissionMetadata = dict[str, JsonValue]
PermissionToolSpec = Any


@dataclass(frozen=True)
class PermissionPolicy:
    """不可变的静态权限规则容器。

    仅存储 rules 和 global_default。
    规则匹配由 StaticPolicyEvaluator 以 last-match-wins 完成。
    """

    rules: tuple[StaticPermission, ...] = ()
    global_default: PermissionDecision | None = None


def _approval_metadata(
    approval_decision: HITLDecision,
    approval_scope: HITLScope,
    *,
    reviewer: str | None = None,
    status: ReviewStatus = "completed",
    rationale: str = "",
    risk: ReviewRisk | None = None,
    authorization: ReviewAuthorization | None = None,
) -> PermissionMetadata:
    metadata: PermissionMetadata = {
        "approval_decision": approval_decision,
        "approval_scope": approval_scope,
    }
    if reviewer is not None:
        metadata["approval_reviewer"] = reviewer
        metadata["approval_review_status"] = status
    if rationale:
        metadata["approval_rationale"] = rationale
    if risk is not None:
        metadata["approval_risk"] = risk
    if authorization is not None:
        metadata["approval_authorization"] = authorization
    return metadata


# ── PermissionEngine — 统一决策引擎 ──

DENIED_BY_USER_GUIDANCE = (
    "; use read-only checks (e.g. git status/git diff) or request manual execution"
)

# 匹配规则来源标识
MATCHED_RESTRICTED_DIRS = "restricted_dirs"
MATCHED_EXECUTION_MODE = "execution_mode"
MATCHED_STATIC_ASK = "static_ask"
MATCHED_SESSION_GRANT = "session_grant"
MATCHED_PERSISTENT_GRANT = "persistent_grant"

MATCHED_DEFAULT = "default"

SOURCE_CONFIG = "config"
SOURCE_SESSION = "session"
SOURCE_PERSISTENT = "persistent"


@dataclass(frozen=True)
class PermissionEngineResult:
    """统一权限决策结果。"""

    decision: PermissionDecision
    blocked: bool
    reason: str = ""
    reason_code: str | None = None
    overrideable: bool | None = None
    remediation: str | None = None
    matched_rule: str | None = None
    source: str | None = None
    metadata: PermissionMetadata | None = None
    approval_result: ApprovalResult | None = None
    action: Action | None = None


def _denial_details(verdict: Verdict) -> tuple[str | None, bool | None, str | None]:
    """从裁决元数据中提取供界面消费的结构化拒绝说明。"""
    reason_code_value = verdict.metadata.get("reason_code")
    overrideable_value = verdict.metadata.get("overrideable")
    remediation_value = verdict.metadata.get("remediation")
    reason_code = reason_code_value if isinstance(reason_code_value, str) else None
    overrideable = overrideable_value if isinstance(overrideable_value, bool) else None
    remediation = remediation_value if isinstance(remediation_value, str) else None
    return reason_code, overrideable, remediation


@dataclass(frozen=True)
class PermissionEngineConfig:
    """PermissionEngine 的静态配置。"""

    static_policy: PermissionPolicy | None = None
    restricted_dirs: tuple[str, ...] = ()
    project_root: Path | None = None
    external_directories: tuple[ExternalDirectory, ...] = ()
    sensitive_path_overrides: tuple[SensitivePathOverride, ...] = ()
    session_grant_store: GrantStore | None = None
    permanent_grant_store: GrantStore | None = None
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = ()
    tool_action_profiles: dict[str, tuple[str, str]] = field(default_factory=dict)
    tool_path_extractors: dict[str, PathExtractor] = field(default_factory=dict)

    # ── 三态 ruleset 支持 ──
    mode_ruleset: tuple[Rule, ...] = ()
    """当前执行模式的默认 ruleset。由 execution_modes.py 在切换 mode 时设置。"""
    user_ruleset: tuple[Rule, ...] = ()
    """用户配置中当前模式的规则 override。从 xcode.config.json 加载。"""
    mode_fallback: PermissionDecision = "ask"
    """未匹配任何规则时的默认决策。plan='deny'，build/act='ask'。"""
    shell_unresolved_policy: ShellUnresolvedPolicy = "ask"
    """Shell 效果无法静态确认时的模式级决策；危险命令始终拒绝。"""
    approval_policy: ApprovalPolicy = "on-request"
    """ask 是否可以交给 reviewer；never 在授权未命中时确定性拒绝。"""


class PermissionEngine:
    """统一权限决策引擎。

    决策优先级（从高到低）：
    0. restricted_dirs、路径边界和危险命令硬阻断
    1. mode、静态策略和 ruleset 取更严格结果
    2. 已保存的 session/permanent grant 处理 ask
    3. approval policy 决定 ask 是否可进入 reviewer
    4. user 或 auto-reviewer 给出最终裁决
    """

    def __init__(self, config: PermissionEngineConfig) -> None:
        self._config = config

    def decide(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        execution_decision: PermissionDecision | None = None,
        tool_spec: PermissionToolSpec | None = None,
        approval_callback: PermissionApprovalCallback | None = None,
        approvals_reviewer: ApprovalsReviewer = "user",
        approval_transcript: str = "",
        approval_turn_id: str = "",
    ) -> PermissionEngineResult:
        profile = self._config.tool_action_profiles.get(tool_name)
        path_extractor = self._config.tool_path_extractors.get(tool_name)
        action = ActionExtractor().extract(
            tool_name,
            tool_input,
            action_profile=profile,
            path_extractor=path_extractor,
        )

        # Tier 0: restricted_dirs 硬阻断
        dir_result = self._check_restricted_dirs(action)
        if dir_result is not None:
            return replace(dir_result, action=action)

        # 统一 resolver 路径：对所有工具生效
        result = self._decide_resolver(
            action,
            execution_decision=execution_decision,
            tool_spec=tool_spec,
            tool_input=tool_input,
            approval_callback=approval_callback,
            approvals_reviewer=approvals_reviewer,
            approval_transcript=approval_transcript,
            approval_turn_id=approval_turn_id,
        )

        # 附加 action 信息到结果
        result = replace(result, action=action)

        return result

    def _has_approval_mechanism(
        self,
        approval_callback: PermissionApprovalCallback | None,
    ) -> bool:
        """检查是否有机制处理 ask 决策。

        如果存在 session grant store、permanent grant store 或 approval_callback
        中的任意一个，则有能力处理 ask。
        """
        return self._config.approval_policy == "never" or (
            self._config.session_grant_store is not None
            or self._config.permanent_grant_store is not None
            or approval_callback is not None
        )

    def _decide_resolver(
        self,
        action: Action,
        *,
        execution_decision: PermissionDecision | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any],
        approval_callback: PermissionApprovalCallback | None = None,
        approvals_reviewer: ApprovalsReviewer = "user",
        approval_transcript: str = "",
        approval_turn_id: str = "",
    ) -> PermissionEngineResult:
        """通过约束求值与 PermissionResolver 生成权限裁决。"""
        verdict = self._policy_verdict(
            action,
            execution_decision=execution_decision,
        )

        if verdict.decision == "deny":
            reason_code, overrideable, remediation = _denial_details(verdict)
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=verdict.reason,
                reason_code=reason_code,
                overrideable=overrideable,
                remediation=remediation,
                matched_rule=verdict.source,
                source=verdict.source,
            )

        if verdict.decision == "allow":
            source = verdict.source
            if source == "mode":
                matched_rule = MATCHED_EXECUTION_MODE
            else:
                matched_rule = MATCHED_DEFAULT
            return PermissionEngineResult(
                decision="allow",
                blocked=False,
                matched_rule=matched_rule,
                source=source,
            )

        # ask → 按工具类型执行授权查找 + 回调
        if not self._has_approval_mechanism(approval_callback):
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason="tool requires approval, no approval mechanism available",
                matched_rule=MATCHED_STATIC_ASK,
            )
        return self._resolve_ask(
            action,
            verdict,
            approval_callback=approval_callback,
            approvals_reviewer=approvals_reviewer,
            tool_spec=tool_spec,
            tool_input=tool_input,
            approval_transcript=approval_transcript,
            approval_turn_id=approval_turn_id,
        )

    def _resolve_ask(
        self,
        action: Action,
        verdict: Verdict,
        *,
        approval_callback: PermissionApprovalCallback | None = None,
        approvals_reviewer: ApprovalsReviewer = "user",
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any],
        approval_transcript: str = "",
        approval_turn_id: str = "",
    ) -> PermissionEngineResult:
        """统一的 ask 处理：grant 查找 + 回调。"""
        return self._resolve_approval_request(
            action,
            verdict,
            approval_callback=approval_callback,
            approvals_reviewer=approvals_reviewer,
            tool_spec=tool_spec,
            tool_input=tool_input,
            approval_transcript=approval_transcript,
            approval_turn_id=approval_turn_id,
        )

    def _policy_verdict(
        self,
        action: Action,
        *,
        execution_decision: PermissionDecision | None,
    ) -> Verdict:
        """把全部策略层转换为约束，再由唯一 resolver 取最严格结果。"""
        policy_constraints = evaluate_policy_constraints(
            action,
            execution_decision=execution_decision,
            static_policy=self._config.static_policy,
            boundary_context=self._boundary_context(),
            hook_constraint_providers=self._config.hook_constraint_providers,
        )

        shell_constraints: tuple[Constraint, ...] = ()
        if action.capability == "shell" and action.unresolved_effects:
            from .shell_analyzer import ShellAnalysisPolicyEvaluator

            shell_constraints = ShellAnalysisPolicyEvaluator().evaluate(
                action,
                self._config.shell_unresolved_policy,
            )

        rules = rule_merge(
            self._config.user_ruleset,
            self._config.mode_ruleset,
        )
        shell_command = self._extract_shell_command(action)
        matched_rule = rule_first_match(
            action,
            rules,
            shell_command=shell_command,
        )
        rule_decision = (
            matched_rule.effect
            if matched_rule is not None
            else self._config.mode_fallback
        )
        rule_metadata: dict[str, object] = {}
        if rule_decision == "deny":
            rule_metadata = {
                "reason_code": "rule_denied",
                "overrideable": False,
                "remediation": "Update the configured permission rule.",
            }
        rule_constraint = Constraint(
            decision=rule_decision,
            source="rule_matcher",
            reason=(
                f"ruleset returned {rule_decision} "
                f"({'matched rule' if matched_rule is not None else 'mode fallback'})"
            ),
            operation=action.operation,
            metadata=rule_metadata,
        )
        return PermissionResolver().resolve(
            policy_constraints + shell_constraints + (rule_constraint,)
        )

    def _extract_shell_command(self, action: Action) -> str | None:
        """从 action 的 targets 中提取 shell 命令字符串。"""
        if action.capability != "shell":
            return None
        for target in action.targets:
            if target.kind == "command":
                return target.value
        return None

    def _boundary_context(self) -> BoundaryContext | None:
        if self._config.project_root is None:
            return None
        return BoundaryContext(
            project_root=self._config.project_root,
            external_directories=self._config.external_directories,
            sensitive_path_overrides=self._config.sensitive_path_overrides,
        )

    # ── 内部检查方法 ──

    def _check_restricted_dirs(
        self,
        action: Action,
    ) -> PermissionEngineResult | None:
        if not self._config.restricted_dirs:
            return None

        path_targets = tuple(
            target for target in action.targets if target.kind == "path"
        )
        for target in path_targets:
            if self._is_restricted_path(target.value):
                return PermissionEngineResult(
                    decision="deny",
                    blocked=True,
                    reason=f"restricted path matched for tool: {action.tool}",
                    reason_code="restricted_directory",
                    overrideable=False,
                    remediation="Remove or adjust the matching restricted_dirs entry.",
                    matched_rule=MATCHED_RESTRICTED_DIRS,
                    source=SOURCE_CONFIG,
                )

        if self._requires_restricted_path_fallback(action, path_targets):
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=(
                    "filesystem paths could not be extracted safely while "
                    f"restricted_dirs is configured for tool: {action.tool}"
                ),
                reason_code="unresolved_path_with_restricted_dirs",
                overrideable=False,
                remediation="Use a command or tool input with a statically resolvable path.",
                matched_rule=MATCHED_RESTRICTED_DIRS,
                source=SOURCE_CONFIG,
            )
        return None

    def _is_restricted_path(self, target_path: str) -> bool:
        """判断结构化路径 target 是否位于任一受限目录内。"""
        project_root = self._config.project_root
        for restricted_dir in self._config.restricted_dirs:
            if project_root is None:
                normalized_target = Path(target_path.replace("\\", "/"))
                normalized_restricted = Path(restricted_dir.replace("\\", "/"))
                if self._path_contains(normalized_target, normalized_restricted):
                    return True
                continue

            resolved_root = project_root.expanduser().resolve(strict=False)
            restricted_path = Path(restricted_dir).expanduser()
            if not restricted_path.is_absolute():
                restricted_path = resolved_root / restricted_path
            target = Path(target_path).expanduser()
            if not target.is_absolute():
                target = resolved_root / target
            try:
                resolved_restricted = restricted_path.resolve(strict=False)
                resolved_target = target.resolve(strict=False)
            except (OSError, RuntimeError):
                return True
            if self._path_contains(resolved_target, resolved_restricted):
                return True
        return False

    def _path_contains(self, candidate: Path, root: Path) -> bool:
        """使用平台路径大小写规则执行目录边界判断。"""
        normalized_candidate = os.path.normcase(os.path.abspath(candidate))
        normalized_root = os.path.normcase(os.path.abspath(root))
        try:
            return os.path.commonpath((normalized_candidate, normalized_root)) == (
                normalized_root
            )
        except ValueError:
            return False

    def _requires_restricted_path_fallback(
        self,
        action: Action,
        path_targets: tuple[Any, ...],
    ) -> bool:
        """判断高风险文件系统输入是否缺少可验证的结构化路径。"""
        if action.tool in {"read_file", "write_file", "edit_file", "apply_patch"}:
            return not path_targets
        if action.capability != "shell" or path_targets:
            return False
        filesystem_commands = {
            "cat",
            "copy-item",
            "cp",
            "del",
            "dir",
            "get-childitem",
            "get-content",
            "head",
            "less",
            "ls",
            "more",
            "move-item",
            "mv",
            "realpath",
            "remove-item",
            "rm",
            "set-content",
            "tail",
        }
        return any(
            target.kind == "command"
            and target.value.split(maxsplit=1)[0].strip("\"'").lower()
            in filesystem_commands
            for target in action.targets
        )

    # ── 统一 ask 处理 ──

    def _resolve_approval_request(
        self,
        action: Action,
        verdict: Verdict,
        *,
        approval_callback: PermissionApprovalCallback | None = None,
        approvals_reviewer: ApprovalsReviewer = "user",
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
        approval_transcript: str = "",
        approval_turn_id: str = "",
    ) -> PermissionEngineResult:
        """执行 ask 后的授权查找、回调调用和授权写入。"""
        candidate = compute_approval_candidate(
            action,
            session_grant_store=self._config.session_grant_store,
            permanent_grant_store=self._config.permanent_grant_store,
            boundary_context=self._boundary_context(),
        )

        # 存在匹配授权 → 直接使用，不回调
        if candidate is not None and candidate.would_resolve != "would_call_approval":
            return self._grant_result(action, candidate)

        # never 不允许交互或模型 reviewer 扩大权限；已存在的显式 grant 仍在上面生效。
        if self._config.approval_policy == "never":
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=f"approval policy is never for tool: {action.tool}",
                reason_code="approval_policy_never",
                overrideable=False,
                remediation=(
                    "Change security.approval_policy to on-request or add an "
                    "explicit allow rule."
                ),
                matched_rule=MATCHED_STATIC_ASK,
                source=SOURCE_CONFIG,
            )

        # 无匹配授权 → 调用 approval_callback
        return self._review_result(
            action,
            verdict,
            approval_callback=approval_callback,
            approvals_reviewer=approvals_reviewer,
            tool_spec=tool_spec,
            tool_input=tool_input,
            approval_transcript=approval_transcript,
            approval_turn_id=approval_turn_id,
        )

    def _grant_result(
        self,
        action: Action,
        candidate: ApprovalCandidate,
    ) -> PermissionEngineResult:
        """授予命中时直接使用授权结果，不调用回调。"""
        winning_grant: GrantRecord | None = None

        for fp in candidate.fingerprints:
            if fp.grant is not None and fp.grant.decision == "deny":
                winning_grant = fp.grant
                break

        if winning_grant is None:
            for fp in candidate.fingerprints:
                if fp.grant is not None and fp.grant.decision == "allow":
                    winning_grant = fp.grant
                    break

        if winning_grant is None:
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason=f"tool requires approval: {action.tool}",
                matched_rule=MATCHED_STATIC_ASK,
            )

        if winning_grant.scope == "session":
            matched_rule = MATCHED_SESSION_GRANT
            source = SOURCE_SESSION
            metadata: PermissionMetadata | None = _approval_metadata(
                winning_grant.decision, winning_grant.scope
            )
        else:
            matched_rule = MATCHED_PERSISTENT_GRANT
            source = SOURCE_PERSISTENT
            metadata = _approval_metadata(winning_grant.decision, winning_grant.scope)

        if winning_grant.decision == "deny":
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=f"permission denied by grant: {winning_grant.grant_id}",
                matched_rule=matched_rule,
                source=source,
                metadata=metadata,
                approval_result=ApprovalResult(
                    decision="deny",
                    scope=winning_grant.scope,
                    grant_id=winning_grant.grant_id,
                ),
            )

        return PermissionEngineResult(
            decision="allow",
            blocked=False,
            matched_rule=matched_rule,
            source=source,
            metadata=metadata,
            approval_result=ApprovalResult(
                decision="allow",
                scope=winning_grant.scope,
                grant_id=winning_grant.grant_id,
            ),
        )

    def _review_result(
        self,
        action: Action,
        verdict: Verdict,
        *,
        approval_callback: PermissionApprovalCallback | None = None,
        approvals_reviewer: ApprovalsReviewer = "user",
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
        approval_transcript: str = "",
        approval_turn_id: str = "",
    ) -> PermissionEngineResult:
        """无匹配授权时调用 approval_callback 并写入新授权存储。"""

        if approval_callback is None or tool_spec is None:
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason=f"tool requires approval: {action.tool}",
                matched_rule=MATCHED_STATIC_ASK,
            )

        request = ApprovalRequest(
            tool=tool_spec,
            action_input=tool_input or {},
            allowed_scopes=_allowed_approval_scopes(action),
            reason=verdict.reason,
            transcript=approval_transcript,
            working_directory=(
                str(self._config.project_root)
                if self._config.project_root is not None
                else ""
            ),
            turn_id=approval_turn_id,
        )
        hitl = approval_callback(request)
        approval_source = (
            "auto_review" if approvals_reviewer == "auto_review" else SOURCE_SESSION
        )

        if hitl.decision == "deny":
            metadata = _approval_metadata(
                "deny",
                hitl.scope,
                reviewer=approvals_reviewer,
                status=hitl.status,
                rationale=hitl.rationale,
                risk=hitl.risk,
                authorization=hitl.authorization,
            )
            if hitl.suggestion:
                metadata = dict(metadata)
                metadata["suggestion"] = hitl.suggestion
            if approvals_reviewer == "auto_review":
                if hitl.status == "timed_out":
                    reason = "automatic approval review timed out"
                elif hitl.status == "failed":
                    reason = "automatic approval review failed closed"
                else:
                    reason = "action rejected by automatic approval review"
            else:
                reason = f"tool {action.tool} denied by user{DENIED_BY_USER_GUIDANCE}"
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=reason,
                matched_rule=MATCHED_STATIC_ASK,
                source=approval_source,
                metadata=metadata,
                approval_result=ApprovalResult(decision="deny", scope=hitl.scope),
            )

        if hitl.scope not in request.allowed_scopes:
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=(
                    f"approval scope {hitl.scope} is not valid for tool: {action.tool}"
                ),
                reason_code="invalid_approval_scope",
                overrideable=False,
                remediation="Retry the approval and select one of the offered scopes.",
                matched_rule=MATCHED_STATIC_ASK,
                source=SOURCE_CONFIG,
            )

        if approvals_reviewer == "auto_review" and hitl.scope != "once":
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason="auto-review may only approve a single execution",
                reason_code="invalid_auto_review_scope",
                overrideable=False,
                remediation="Retry auto-review with once scope or ask the user.",
                matched_rule=MATCHED_STATIC_ASK,
                source=SOURCE_CONFIG,
            )

        # 允许 — 根据实际展示并选择的 scope 写入授权
        metadata: PermissionMetadata = _approval_metadata(
            "allow",
            hitl.scope,
            reviewer=approvals_reviewer,
            status=hitl.status,
            rationale=hitl.rationale,
            risk=hitl.risk,
            authorization=hitl.authorization,
        )
        write_scope = hitl.scope
        if write_scope == "session":
            self._write_grants(action, decision="allow", scope="session")
        elif write_scope == "permanent":
            self._write_grants(action, decision="allow", scope="permanent")

        return PermissionEngineResult(
            decision="allow",
            blocked=False,
            matched_rule=MATCHED_STATIC_ASK,
            source=approval_source,
            metadata=metadata,
            approval_result=ApprovalResult(
                decision="allow",
                scope=hitl.scope,
            ),
        )

    def _write_grants(
        self,
        action: Action,
        *,
        decision: _GrantDecision,
        scope: _GrantScope,
    ) -> None:
        """为 action 的每个 target 写入结构化授权记录。"""
        store: GrantStore | None = None
        if scope == "session":
            store = self._config.session_grant_store
        elif scope == "permanent":
            store = self._config.permanent_grant_store

        if store is None:
            return

        for target in action.targets:
            grant = create_grant_record(action, target, decision=decision, scope=scope)
            store.add(grant)


PermissionCheckResult = PermissionEngineResult


def _allowed_approval_scopes(action: Action) -> tuple[ApprovalScope, ...]:
    """返回当前动作可真实持久化的审批范围。"""
    if len(action.targets) > 1:
        return ("once",)
    if action.capability == "unknown":
        return ("once", "session")
    return ("once", "session", "permanent")
