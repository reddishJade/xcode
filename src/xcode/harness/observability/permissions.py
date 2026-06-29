"""工具执行的 allow/deny/ask 权限策略与 HITL 授权模型。

权限架构：静态策略（规则 + global_default）+ 动态策略（执行模式）+ HITL 授权（会话/持久）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Literal

from ..session import JsonValue
from .permission_model import (
    Action,
    ActionExtractor,
    ApprovalCandidate,
    ApprovalResult,
    BoundaryContext,
    ExternalDirectory,
    GrantRecord,
    GrantStore,
    PolicyEvaluator,
    PermissionResolver,
    StaticPermission,
    StructuredBoundaryPolicyEvaluator,
    Verdict,
    compute_shadow_approval_candidate,
    create_grant_record,
    evaluate_policy_constraints,
)
from .permission_model import GrantDecision as _GrantDecision
from .permission_model import GrantScope as _GrantScope


PermissionDecision = Literal["allow", "deny", "ask"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]
type PermissionMetadata = dict[str, JsonValue]


@dataclass(frozen=True)
class HITLResult:
    """用户对工具授权的结构化结果。"""

    decision: HITLDecision
    scope: HITLScope


PermissionToolSpec = Any
PermissionApprovalCallback = Callable[[Any, dict[str, Any]], HITLResult]


class PermissionPolicy:
    """不可变的静态权限规则容器。

    仅存储 rules 和 global_default。
    规则匹配由 StaticPolicyEvaluator 以 last-match-wins 完成。
    """

    def __init__(
        self,
        rules: tuple[StaticPermission, ...] = (),
        global_default: str | None = None,
    ) -> None:
        self.rules = rules
        self.global_default = global_default


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
    matched_rule: str | None = None
    source: str | None = None
    metadata: PermissionMetadata | None = None
    shadow_action: Action | None = None
    shadow_verdict: Verdict | None = None
    shadow_diff: str | None = None
    shadow_approval_candidate: ApprovalCandidate | None = None
    approval_result: ApprovalResult | None = None
    action: Action | None = None


@dataclass(frozen=True)
class PermissionEngineConfig:
    """PermissionEngine 的静态配置。"""

    static_policy: PermissionPolicy | None = None
    restricted_dirs: tuple[str, ...] = ()
    shadow_model_enabled: bool = False
    project_root: Path | None = None
    external_directories: tuple[ExternalDirectory, ...] = ()
    session_grant_store: GrantStore | None = None
    permanent_grant_store: GrantStore | None = None
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = ()
    tool_action_profiles: dict[str, tuple[str, str]] = field(default_factory=dict)


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
        action = ActionExtractor().extract(
            tool_name, tool_input, action_profile=profile
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
            metadata: PermissionMetadata | None = None
            winning = verdict.winning_constraint
            if winning is not None and winning.non_bypassable:
                metadata = {"non_bypassable": True}
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=verdict.reason,
                matched_rule=verdict.source,
                source=verdict.source,
                metadata=metadata,
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
        """解析当前已接入的 shadow policy constraints。"""
        constraints = evaluate_policy_constraints(
            action,
            execution_decision=execution_decision,
            static_policy=self._config.static_policy,
            boundary_context=self._boundary_context(),
            safety_backstop_enabled=True,
            hook_constraint_providers=self._config.hook_constraint_providers,
        )
        return PermissionResolver().resolve(constraints)

    def _compute_shadow_approval(
        self,
        action: Action,
        verdict: Verdict,
    ) -> ApprovalCandidate | None:
        """当 shadow verdict 为 ask 时，预测 engine-level grant/callback 结果。"""
        if action.tool not in StructuredBoundaryPolicyEvaluator.STRUCTURED_TOOLS:
            return None
        if verdict.decision != "ask":
            return None

        return compute_shadow_approval_candidate(
            action,
            session_grant_store=self._config.session_grant_store,
            permanent_grant_store=self._config.permanent_grant_store,
            boundary_context=self._boundary_context(),
        )

    def _boundary_context(self) -> BoundaryContext | None:
        if self._config.project_root is None:
            return None
        return BoundaryContext(
            project_root=self._config.project_root,
            external_directories=self._config.external_directories,
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
                    matched_rule=MATCHED_RESTRICTED_DIRS,
                    source=SOURCE_CONFIG,
                )

        if self._requires_restricted_path_fallback(action, path_targets):
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason=(
                    "filesystem paths could not be extracted safely while "
                    f"restricted_dirs is configured for tool: {action.tool}"
                ),
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
        """对无法结构化解析的高风险文件系统输入采用 ask。"""
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
        is_multi_target = len(action.targets) > 1

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
            is_multi_target=is_multi_target,
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
        is_multi_target: bool = False,
    ) -> PermissionEngineResult:
        """无匹配授权时调用 approval_callback 并写入新授权存储。"""

        if approval_callback is None or tool_spec is None:
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason=f"tool requires approval: {action.tool}",
                matched_rule=MATCHED_STATIC_ASK,
            )

        hitl = approval_callback(tool_spec, tool_input or {})

        if hitl.decision == "deny":
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=(f"tool {action.tool} denied by user{DENIED_BY_USER_GUIDANCE}"),
                matched_rule=MATCHED_STATIC_ASK,
                source=SOURCE_SESSION,
                metadata=_approval_metadata("deny", hitl.scope),
                approval_result=ApprovalResult(decision="deny", scope=hitl.scope),
            )

        # 允许 — 根据 scope 写入授权
        effective_scope = hitl.scope
        metadata: PermissionMetadata = _approval_metadata("allow", hitl.scope)

        # unknown-tool tools are session/once only — downgrade permanent
        if action.capability == "unknown" and hitl.scope == "permanent":
            effective_scope = "session"
            metadata = dict(metadata)
            metadata["requested_scope"] = "permanent"
            metadata["effective_scope"] = "session"
            metadata["capability_scope_restriction"] = True

        if is_multi_target and hitl.scope in ("session", "permanent"):
            effective_scope = "once"
            metadata = dict(metadata)
            metadata["requested_scope"] = hitl.scope
            metadata["effective_scope"] = "once"
            metadata["multi_target_restriction"] = True

        write_scope = effective_scope
        if write_scope == "session" and not is_multi_target:
            self._write_grants(action, decision="allow", scope="session")
        elif write_scope == "permanent" and not is_multi_target:
            self._write_grants(action, decision="allow", scope="permanent")

        return PermissionEngineResult(
            decision="allow",
            blocked=False,
            matched_rule=MATCHED_STATIC_ASK,
            source=SOURCE_SESSION,
            metadata=metadata,
            approval_result=ApprovalResult(
                decision="allow",
                scope=effective_scope,
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
