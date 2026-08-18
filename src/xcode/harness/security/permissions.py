"""工具执行的 allow/deny/ask 权限策略与 HITL 授权模型。

权限架构：静态策略（规则 + global_default）+ 动态策略（执行模式）+ HITL 授权（会话/持久）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Literal

from xcode.agent.types import ApprovalRequest, ApprovalScope

from ..session import JsonValue
from .permission_model import (
    Action,
    ActionExtractor,
    ApprovalCandidate,
    ApprovalResult,
    BoundaryContext,
    Constraint,
    ExternalDirectory,
    GrantRecord,
    GrantStore,
    PolicyEvaluator,
    PermissionResolver,
    PathExtractor,
    Rule,
    SensitivePathOverride,
    StaticPermission,
    Verdict,
    compute_shadow_approval_candidate,
    create_grant_record,
)
from .rule_matcher import first_match as rule_first_match, merge_rulesets as rule_merge
from .permission_model import GrantDecision as _GrantDecision
from .permission_model import GrantScope as _GrantScope


PermissionDecision = Literal["allow", "deny", "ask"]
ShellUnresolvedPolicy = Literal["allow", "deny", "ask"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]
type PermissionMetadata = dict[str, JsonValue]


@dataclass(frozen=True)
class HITLResult:
    """用户对工具授权的结构化结果。"""

    decision: HITLDecision
    scope: HITLScope
    suggestion: str = ""


PermissionToolSpec = Any
PermissionApprovalCallback = Callable[[ApprovalRequest], HITLResult]


@dataclass(frozen=True)
class PermissionPolicy:
    """不可变的静态权限规则容器。

    仅存储 rules 和 global_default。
    规则匹配由 StaticPolicyEvaluator 以 last-match-wins 完成。
    """

    rules: tuple[StaticPermission, ...] = ()
    global_default: PermissionDecision | None = None


def _approval_metadata(
    user_decision: HITLDecision, approval_scope: HITLScope
) -> PermissionMetadata:
    return {
        "user_decision": user_decision,
        "approval_scope": approval_scope,
    }


# ── PermissionEngine — 统一决策引擎 ──

DENIED_BY_USER_GUIDANCE = (
    "; use read-only checks (e.g. git status/git diff) or request manual execution"
)

# 匹配规则来源标识
MATCHED_RESTRICTED_DIRS = "restricted_dirs"
MATCHED_STATIC_DENY = "static_deny"
MATCHED_EXECUTION_MODE = "execution_mode"
MATCHED_STATIC_ASK = "static_ask"
MATCHED_SESSION_GRANT = "session_grant"
MATCHED_PERSISTENT_GRANT = "persistent_grant"
MATCHED_STATIC_ALLOW = "static_allow"

MATCHED_DEFAULT = "default"

SOURCE_CONFIG = "config"
SOURCE_SESSION = "session"
SOURCE_PERSISTENT = "persistent"

SOURCE_EXECUTION_MODE = "execution_mode"
SOURCE_DEFAULT = "default"


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
    shadow_action: Action | None = None
    shadow_verdict: Verdict | None = None
    shadow_diff: str | None = None
    shadow_approval_candidate: ApprovalCandidate | None = None
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
    shadow_model_enabled: bool = False
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
    """未匹配任何规则时的默认决策。plan='deny', build='allow', act='ask'。"""
    shell_unresolved_policy: ShellUnresolvedPolicy = "ask"
    """Shell 效果无法静态确认时的模式级决策；危险命令始终拒绝。"""


class PermissionEngine:
    """统一权限决策引擎。

    决策优先级（从高到低）：
    0. restricted_dirs 硬阻断
    1. 静态 deny > 执行模式 deny
    2. 静态 ask
    3. HITL 授权（session/persistent 满足前面的 ask）
    4. 静态 allow
    5. <removed: risk_evaluator / high_risk 见 section 8>
    6. 高风险审批
    7. 默认放行
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
        )

        # 附加 action 信息到结果
        result = replace(result, action=action)

        # Shadow 模式：附加 shadow 信息到结果
        if not self._config.shadow_model_enabled:
            return result

        shadow_verdict = self._shadow_verdict(
            action,
            execution_decision=execution_decision,
        )
        shadow_approval_candidate = self._compute_shadow_approval(
            action,
            shadow_verdict,
        )
        return replace(
            result,
            shadow_action=action,
            shadow_verdict=shadow_verdict,
            shadow_diff=self._shadow_diff(result, shadow_verdict),
            shadow_approval_candidate=shadow_approval_candidate,
        )

    def _has_approval_mechanism(
        self,
        approval_callback: PermissionApprovalCallback | None,
    ) -> bool:
        """检查是否有机制处理 ask 决策。

        如果存在 session grant store、permanent grant store 或 approval_callback
        中的任意一个，则有能力处理 ask。
        """
        return (
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
    ) -> PermissionEngineResult:
        """通过约束求值与 PermissionResolver 生成权限裁决。"""
        verdict = self._shadow_verdict(
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
            tool_spec=tool_spec,
            tool_input=tool_input,
        )

    def _resolve_ask(
        self,
        action: Action,
        verdict: Verdict,
        *,
        approval_callback: PermissionApprovalCallback | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any],
    ) -> PermissionEngineResult:
        """统一的 ask 处理：grant 查找 + 回调。"""
        # 统一路径：所有工具通过 _execute_cutover_ask 处理 grant 查找、回调、写入
        return self._execute_cutover_ask(
            action,
            verdict,
            approval_callback=approval_callback,
            tool_spec=tool_spec,
            tool_input=tool_input,
        )

    def _shadow_verdict(
        self,
        action: Action,
        *,
        execution_decision: PermissionDecision | None,
    ) -> Verdict:
        """通过规则匹配生成权限裁决。

        使用 RuleMatcher 对 action 做规则匹配。
        PathBoundary 安全网和 ModePolicy 仍然保留。
        """
        # Step 1: PathBoundary 安全网
        from .permission_model import PathBoundaryPolicyEvaluator

        ctx = self._boundary_context()
        boundary_constraints: tuple[Constraint, ...] = PathBoundaryPolicyEvaluator(
            ctx
        ).evaluate(action)
        for c in boundary_constraints:
            if c.decision == "deny":
                return Verdict(
                    decision="deny",
                    source="boundary",
                    reason=c.reason,
                    winning_constraint=c,
                    constraints=(c,),
                    metadata=c.metadata,
                )

        # Step 2: ModePolicy 仍保留（execution_decision 来自 check_call）
        mode_constraints: tuple[Constraint, ...] = ()
        if execution_decision is not None:
            from .permission_model import ModePolicyEvaluator

            mode_constraints = ModePolicyEvaluator(execution_decision).evaluate(action)
            for c in mode_constraints:
                if c.decision == "deny":
                    return Verdict(
                        decision="deny",
                        source="mode",
                        reason=c.reason,
                        winning_constraint=c,
                        constraints=(c,),
                        metadata=c.metadata,
                    )

        # Step 3: StaticPolicy（用户配置的静态规则）
        static_constraints: tuple[Constraint, ...] = ()
        if self._config.static_policy is not None:
            from .permission_model import StaticPolicyEvaluator

            static_constraints = StaticPolicyEvaluator(
                self._config.static_policy.rules,
                global_default=self._config.static_policy.global_default,
            ).evaluate(action)

        # Step 4: ShellAnalysis 不可解析效果按当前执行模式处理
        shell_constraints: tuple[Constraint, ...] = ()
        if action.capability == "shell" and action.unresolved_effects:
            from .shell_analyzer import ShellAnalysisPolicyEvaluator

            shell_constraints = ShellAnalysisPolicyEvaluator().evaluate(
                action,
                self._config.shell_unresolved_policy,
            )

        # Step 5: RuleMatcher 三态 ruleset 匹配
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

        # Step 6: 合并约束
        all_constraints = (
            mode_constraints
            + static_constraints
            + boundary_constraints
            + shell_constraints
        )
        resolver_verdict = PermissionResolver().resolve(all_constraints)

        # 合并 RuleMatcher 与 resolver：
        #   RuleMatcher 提供 mode/user 规则决策（主决策源）
        #   Resolver 处理 PathBoundary/Mode/Static 约束（安全网 + 静态规则）
        #   两者冲突时，较严格的一方获胜
        if matched_rule is not None:
            rule_decision = matched_rule.effect
        else:
            rule_decision = self._config.mode_fallback

        # 取较严格的决策：deny > ask > allow
        decision_priority = {"allow": 0, "ask": 1, "deny": 2}
        final_decision = resolver_verdict.decision
        if decision_priority[rule_decision] > decision_priority[final_decision]:
            final_decision = rule_decision

        if final_decision != resolver_verdict.decision:
            metadata: dict[str, object] = {}
            if final_decision == "deny":
                metadata = {
                    "reason_code": "rule_denied",
                    "overrideable": False,
                    "remediation": "Update the configured permission rule.",
                }
            return Verdict(
                decision=final_decision,
                source="rule_matcher",
                reason=(
                    f"rule={rule_decision}, resolver={resolver_verdict.decision} "
                    f"({'matched rule' if matched_rule else 'fallback'})"
                ),
                winning_constraint=None,
                constraints=all_constraints,
                metadata=metadata,
            )

        return resolver_verdict

    def _compute_shadow_approval(
        self,
        action: Action,
        verdict: Verdict,
    ) -> ApprovalCandidate | None:
        """当 shadow verdict 为 ask 时，预测 engine-level grant/callback 结果。"""
        if not action.targets:
            return None
        if verdict.decision != "ask":
            return None

        return compute_shadow_approval_candidate(
            action,
            session_grant_store=self._config.session_grant_store,
            permanent_grant_store=self._config.permanent_grant_store,
            boundary_context=self._boundary_context(),
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

    def _shadow_diff(
        self,
        current_result: PermissionEngineResult,
        shadow_verdict: Verdict,
    ) -> str | None:
        if current_result.decision == shadow_verdict.decision:
            return None
        return (
            "current decision "
            f"{current_result.decision} differs from shadow decision "
            f"{shadow_verdict.decision}"
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

    def _execute_cutover_ask(
        self,
        action: Action,
        verdict: Verdict,
        *,
        approval_callback: PermissionApprovalCallback | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
    ) -> PermissionEngineResult:
        """执行 ask 后的授权查找、回调调用和授权写入。"""
        candidate = compute_shadow_approval_candidate(
            action,
            session_grant_store=self._config.session_grant_store,
            permanent_grant_store=self._config.permanent_grant_store,
            boundary_context=self._boundary_context(),
        )

        # 存在匹配授权 → 直接使用，不回调
        if candidate is not None and candidate.would_resolve != "would_call_approval":
            return self._cutover_grant_result(action, candidate)

        # 无匹配授权 → 调用 approval_callback
        return self._cutover_callback_result(
            action,
            verdict,
            approval_callback=approval_callback,
            tool_spec=tool_spec,
            tool_input=tool_input,
        )

    def _cutover_grant_result(
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

    def _cutover_callback_result(
        self,
        action: Action,
        verdict: Verdict,
        *,
        approval_callback: PermissionApprovalCallback | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
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
        )
        hitl = approval_callback(request)

        if hitl.decision == "deny":
            metadata = _approval_metadata("deny", hitl.scope)
            if hitl.suggestion:
                metadata = dict(metadata)
                metadata["suggestion"] = hitl.suggestion
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=(f"tool {action.tool} denied by user{DENIED_BY_USER_GUIDANCE}"),
                matched_rule=MATCHED_STATIC_ASK,
                source=SOURCE_SESSION,
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

        # 允许 — 根据实际展示并选择的 scope 写入授权
        metadata: PermissionMetadata = _approval_metadata("allow", hitl.scope)
        write_scope = hitl.scope
        if write_scope == "session":
            self._write_grants(action, decision="allow", scope="session")
        elif write_scope == "permanent":
            self._write_grants(action, decision="allow", scope="permanent")

        return PermissionEngineResult(
            decision="allow",
            blocked=False,
            matched_rule=MATCHED_STATIC_ASK,
            source=SOURCE_SESSION,
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
