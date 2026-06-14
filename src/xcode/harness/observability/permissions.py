"""工具执行的 allow/deny/ask 权限策略与 HITL 授权模型。

权限架构：静态策略（配置） + 动态策略（执行模式）+ HITL 授权（会话/持久）

决策优先级（从高到低）：
0. restricted_dirs — 目录限制，硬阻断
1. 静态 deny — SecurityRuntimeConfig.deny_tools
2. 执行模式 deny — Plan/Review 模式禁用
3. 静态 ask — SecurityRuntimeConfig.ask_tools
4. HITL 授权 — session/persistent 满足前面的 ask
    5. 静态 allow — SecurityRuntimeConfig.allow_tools
    6. 默认放行 (risk_evaluator / high_risk 已移除)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Literal, cast

from ..session import JsonValue
from .permission_model import (
    Action,
    ActionExtractor,
    ApprovalCandidate,
    ApprovalResult,
    BoundaryContext,
    GrantRecord,
    GrantStore,
    PolicyEvaluator,
    PermissionResolver,
    StructuredBoundaryPolicyEvaluator,
    UnmappableLegacyGrant,
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
type PermissionRuleData = dict[str, JsonValue]


@dataclass(frozen=True)
class HITLResult:
    """用户对工具授权的结构化结果。"""

    decision: HITLDecision
    scope: HITLScope


@dataclass(frozen=True)
class PermissionRule:
    tool: str
    decision: PermissionDecision
    input_contains: str | None = None
    input_prefix: str | None = None


PermissionToolSpec = Any
PermissionApprovalCallback = Callable[[Any, dict[str, Any]], HITLResult]


class PermissionPolicy:
    """不可变的静态权限规则集，来自配置。

    decide() 按 deny > ask > allow 优先级返回，不依赖规则插入顺序。
    """

    def __init__(self, rules: tuple[PermissionRule, ...] = ()) -> None:
        self.rules = rules

    def decide(self, tool_name: str, action_input: str) -> PermissionDecision | None:
        matching: list[PermissionDecision] = []
        for rule in self.rules:
            if rule.tool != tool_name and rule.tool != "*":
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
            matching.append(rule.decision)
        if "deny" in matching:
            return "deny"
        if "ask" in matching:
            return "ask"
        if "allow" in matching:
            return "allow"
        return None


class SessionPermissionPolicy:
    """运行时会话级权限覆写。grant() 替换同 key 的旧规则，最后添加的规则优先。"""

    def __init__(self) -> None:
        self._rules: list[PermissionRule] = []

    def grant(
        self,
        tool_name: str,
        decision: PermissionDecision,
        input_contains: str | None = None,
        input_prefix: str | None = None,
    ) -> None:
        matching_key = (tool_name, input_contains, input_prefix)
        self._rules = [
            r
            for r in self._rules
            if (r.tool, r.input_contains, r.input_prefix) != matching_key
        ]
        self._rules.append(
            PermissionRule(tool_name, decision, input_contains, input_prefix)
        )

    def decide(self, tool_name: str, action_input: str) -> PermissionDecision | None:
        result: PermissionDecision | None = None
        for rule in self._rules:
            if rule.tool != tool_name and rule.tool != "*":
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
            result = rule.decision
        return result

    @property
    def rules(self) -> list[PermissionRule]:
        return list(self._rules)

    def clear(self) -> None:
        self._rules.clear()


class PersistentPermissionStore:
    """文件持久化的授权规则，写入 .local/hitl_policy.json。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> PermissionPolicy:
        if not self.path.exists():
            return PermissionPolicy()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return PermissionPolicy()
            rules = [
                rule
                for item in raw
                if (rule := _permission_rule_from_data(item)) is not None
            ]
            return PermissionPolicy(tuple(rules))
        except (OSError, json.JSONDecodeError):
            return PermissionPolicy()

    def grant(
        self,
        tool_name: str,
        decision: PermissionDecision,
        input_contains: str | None = None,
        input_prefix: str | None = None,
    ) -> PermissionPolicy:
        current = self.load()
        new_rule = PermissionRule(tool_name, decision, input_contains, input_prefix)
        filtered = tuple(
            r
            for r in current.rules
            if not (
                r.tool == tool_name
                and r.input_contains == input_contains
                and r.input_prefix == input_prefix
            )
        )
        updated = filtered + (new_rule,)
        self._write(updated)
        return PermissionPolicy(updated)

    def revoke(
        self, tool_name: str, input_contains: str | None = None
    ) -> PermissionPolicy:
        current = self.load()
        updated = tuple(
            r
            for r in current.rules
            if not (r.tool == tool_name and r.input_contains == input_contains)
        )
        self._write(updated)
        return PermissionPolicy(updated)

    def _write(self, rules: tuple[PermissionRule, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: list[PermissionRuleData] = []
        for r in rules:
            entry: PermissionRuleData = {"tool": r.tool, "decision": r.decision}
            if r.input_contains is not None:
                entry["input_contains"] = r.input_contains
            if r.input_prefix is not None:
                entry["input_prefix"] = r.input_prefix
            data.append(entry)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ── 辅助函数 ──


def _permission_rule_from_data(value: object) -> PermissionRule | None:
    if not isinstance(value, dict):
        return None
    tool = value.get("tool")
    decision = value.get("decision")
    if not isinstance(tool, str) or not tool:
        return None
    if decision not in ("allow", "deny", "ask"):
        return None
    input_contains = value.get("input_contains")
    input_prefix = value.get("input_prefix")
    return PermissionRule(
        tool=tool,
        decision=decision,
        input_contains=input_contains if isinstance(input_contains, str) else None,
        input_prefix=input_prefix if isinstance(input_prefix, str) else None,
    )


def _approval_metadata(
    user_decision: HITLDecision, approval_scope: HITLScope
) -> PermissionMetadata:
    return {
        "user_decision": user_decision,
        "approval_scope": approval_scope,
    }


def _attach_unmappable_legacy_grants(
    metadata: PermissionMetadata | None,
    unmappable_grants: tuple[UnmappableLegacyGrant, ...],
) -> PermissionMetadata:
    """当存在无法映射的 legacy 授权时将其附加到 metadata。"""
    if not unmappable_grants:
        return metadata or {}
    result = dict(metadata) if metadata is not None else {}
    result["unmappable_legacy_grants"] = [
        {
            "tool": g.tool,
            "decision": g.decision,
            "reason": g.reason,
        }
        for g in unmappable_grants
    ]
    return result


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

MATCHED_LEGACY_ADAPTER = "legacy_adapter"
MATCHED_DEFAULT = "default"

SOURCE_CONFIG = "config"
SOURCE_SESSION = "session"
SOURCE_PERSISTENT = "persistent"

SOURCE_EXECUTION_MODE = "execution_mode"
SOURCE_LEGACY = "legacy"
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


@dataclass(frozen=True)
class PermissionEngineConfig:
    """PermissionEngine 的静态配置。"""

    static_policy: PermissionPolicy | None = None
    session_policy: SessionPermissionPolicy | None = None
    persistent_store: PersistentPermissionStore | None = None
    restricted_dirs: tuple[str, ...] = ()
    allowlist_mode: bool = False
    defer_static_ask: bool = False
    shadow_model_enabled: bool = False
    project_root: Path | None = None
    session_grant_store: GrantStore | None = None
    permanent_grant_store: GrantStore | None = None
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = ()


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
        action_input: str,
        *,
        execution_decision: PermissionDecision | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
        approval_callback: PermissionApprovalCallback | None = None,
    ) -> PermissionEngineResult:
        # Tier 0: restricted_dirs 硬阻断
        dir_result = self._check_restricted_dirs(action_input, tool_name)
        if dir_result is not None:
            return dir_result

        # 统一 resolver 路径：对所有工具生效
        result = self._decide_resolver(
            tool_name,
            action_input,
            execution_decision=execution_decision,
            tool_spec=tool_spec,
            tool_input=tool_input,
            approval_callback=approval_callback,
        )

        # Shadow 模式：附加 shadow 信息到结果
        if not self._config.shadow_model_enabled:
            return result

        shadow_action = ActionExtractor().extract(tool_name, tool_input or {})
        shadow_verdict = self._shadow_verdict(
            shadow_action,
            action_input=action_input,
            execution_decision=execution_decision,
        )
        shadow_approval_candidate = self._compute_shadow_approval(
            shadow_action,
            shadow_verdict,
        )
        return replace(
            result,
            shadow_action=shadow_action,
            shadow_verdict=shadow_verdict,
            shadow_diff=self._shadow_diff(result, shadow_verdict),
            shadow_approval_candidate=shadow_approval_candidate,
        )

    def _decide_resolver(
        self,
        tool_name: str,
        action_input: str,
        *,
        execution_decision: PermissionDecision | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
        approval_callback: PermissionApprovalCallback | None = None,
    ) -> PermissionEngineResult:
        """统一的 resolver 决策路径（替换 _decide_current / _decide_cutover / _decide_shell_cutover）。"""
        action = ActionExtractor().extract(tool_name, tool_input or {})
        verdict = self._shadow_verdict(
            action,
            action_input=action_input,
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
        if self._config.defer_static_ask:
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason="tool requires approval",
                matched_rule=MATCHED_STATIC_ASK,
            )
        return self._resolve_ask(
            action,
            verdict,
            action_input=action_input,
            approval_callback=approval_callback,
            tool_spec=tool_spec,
            tool_input=tool_input,
        )

    def _resolve_ask(
        self,
        action: Action,
        verdict: Verdict,
        *,
        action_input: str,
        approval_callback: PermissionApprovalCallback | None = None,
        tool_spec: PermissionToolSpec | None = None,
        tool_input: dict[str, Any] | None = None,
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
        action_input: str,
        execution_decision: PermissionDecision | None,
    ) -> Verdict:
        """解析当前已接入的 shadow policy constraints。"""
        constraints = evaluate_policy_constraints(
            action,
            execution_decision=execution_decision,
            static_policy=self._config.static_policy,
            allowlist_mode=self._config.allowlist_mode,
            action_input=action_input,
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

        legacy_sources: list[object] = []
        if self._config.static_policy is not None:
            legacy_sources.append(self._config.static_policy)
        if self._config.session_policy is not None:
            legacy_sources.append(self._config.session_policy)
        if self._config.persistent_store is not None:
            legacy_sources.append(self._config.persistent_store)

        return compute_shadow_approval_candidate(
            action,
            session_grant_store=self._config.session_grant_store,
            permanent_grant_store=self._config.permanent_grant_store,
            legacy_sources=tuple(legacy_sources),
            boundary_context=self._boundary_context(),
        )

    def _boundary_context(self) -> BoundaryContext | None:
        if self._config.project_root is None:
            return None
        return BoundaryContext(self._config.project_root)

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
        action_input: str,
        tool_name: str,
    ) -> PermissionEngineResult | None:
        input_lower = action_input.lower()
        for restricted_dir in self._config.restricted_dirs:
            if restricted_dir.lower() in input_lower:
                return PermissionEngineResult(
                    decision="deny",
                    blocked=True,
                    reason=f"restricted directory matched for tool: {tool_name}",
                    matched_rule=MATCHED_RESTRICTED_DIRS,
                    source=SOURCE_CONFIG,
                )
        return None

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

        # 收集 legacy 授权源（与 shadow 路径共用）
        legacy_sources: list[object] = []
        if self._config.static_policy is not None:
            legacy_sources.append(self._config.static_policy)
        if self._config.session_policy is not None:
            legacy_sources.append(self._config.session_policy)
        if self._config.persistent_store is not None:
            legacy_sources.append(self._config.persistent_store)

        candidate = compute_shadow_approval_candidate(
            action,
            session_grant_store=self._config.session_grant_store,
            permanent_grant_store=self._config.permanent_grant_store,
            legacy_sources=tuple(legacy_sources),
            boundary_context=self._boundary_context(),
        )

        # 存在匹配授权 → 直接使用，不回调
        if candidate is not None and candidate.would_resolve != "would_call_approval":
            return self._cutover_grant_result(
                action,
                candidate,
                unmappable_legacy_grants=candidate.unmappable_legacy_grants,
            )

        unmappable_grants = (
            candidate.unmappable_legacy_grants if candidate is not None else ()
        )

        # 无匹配授权 → 调用 approval_callback
        return self._cutover_callback_result(
            action,
            verdict,
            approval_callback=approval_callback,
            tool_spec=tool_spec,
            tool_input=tool_input,
            is_multi_target=is_multi_target,
            unmappable_legacy_grants=unmappable_grants,
        )

    def _cutover_grant_result(
        self,
        action: Action,
        candidate: ApprovalCandidate,
        *,
        unmappable_legacy_grants: tuple[UnmappableLegacyGrant, ...] = (),
    ) -> PermissionEngineResult:
        """授予命中时直接使用授权结果，不调用回调。"""
        winning_grant: GrantRecord | None = None
        is_legacy = False

        for fp in candidate.fingerprints:
            if fp.grant is not None and fp.grant.decision == "deny":
                winning_grant = fp.grant
                is_legacy = fp.source == "legacy_adapter"
                break

        if winning_grant is None:
            for fp in candidate.fingerprints:
                if fp.grant is not None and fp.grant.decision == "allow":
                    winning_grant = fp.grant
                    is_legacy = fp.source == "legacy_adapter"
                    break

        if winning_grant is None:
            return PermissionEngineResult(
                decision="ask",
                blocked=True,
                reason=f"tool requires approval: {action.tool}",
                matched_rule=MATCHED_STATIC_ASK,
            )

        if is_legacy:
            matched_rule = MATCHED_LEGACY_ADAPTER
            source = SOURCE_LEGACY
            metadata: PermissionMetadata | None = {"legacy_adapter": True}
        elif winning_grant.scope == "session":
            matched_rule = MATCHED_SESSION_GRANT
            source = SOURCE_SESSION
            metadata = _approval_metadata(winning_grant.decision, winning_grant.scope)
        else:
            matched_rule = MATCHED_PERSISTENT_GRANT
            source = SOURCE_PERSISTENT
            metadata = _approval_metadata(winning_grant.decision, winning_grant.scope)

        metadata = _attach_unmappable_legacy_grants(metadata, unmappable_legacy_grants)

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
        unmappable_legacy_grants: tuple[UnmappableLegacyGrant, ...] = (),
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
                metadata=_attach_unmappable_legacy_grants(
                    _approval_metadata("deny", hitl.scope),
                    unmappable_legacy_grants,
                ),
                approval_result=ApprovalResult(decision="deny", scope=hitl.scope),
            )

        # 允许 — 根据 scope 写入授权
        effective_scope = hitl.scope
        metadata: PermissionMetadata = _attach_unmappable_legacy_grants(
            _approval_metadata("allow", hitl.scope),
            unmappable_legacy_grants,
        )

        if is_multi_target and hitl.scope in ("session", "permanent"):
            effective_scope = "once"
            metadata = dict(metadata)
            metadata["requested_scope"] = hitl.scope
            metadata["effective_scope"] = "once"
            metadata["multi_target_restriction"] = True

        if hitl.scope == "session" and not is_multi_target:
            self._write_grants(action, decision="allow", scope="session")
        elif hitl.scope == "permanent" and not is_multi_target:
            self._write_grants(action, decision="allow", scope="permanent")

        return PermissionEngineResult(
            decision="allow",
            blocked=False,
            matched_rule=MATCHED_STATIC_ASK,
            source=SOURCE_SESSION,
            metadata=metadata,
            approval_result=ApprovalResult(
                decision="allow",
                scope=cast(Literal["once", "session", "permanent"], effective_scope),
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
