"""工具执行的 allow/deny/ask 权限策略与 HITL 授权模型。

权限架构：静态策略（配置） + 动态策略（执行模式）+ HITL 授权（会话/持久）

决策优先级（从高到低）：
0. restricted_dirs — 目录限制，硬阻断
1. 静态 deny — SecurityRuntimeConfig.deny_tools
2. 执行模式 deny — Plan/Review 模式禁用
3. 静态 ask — SecurityRuntimeConfig.ask_tools
4. HITL 授权 — session/persistent 满足前面的 ask
5. 静态 allow — SecurityRuntimeConfig.allow_tools
6. risk_evaluator deny/ask/allow — 工具运行时风险决策
7. 高风险审批 — tool.risk=="high" 且 approval_policy 要求审批
8. 默认放行
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Protocol

from ..session import JsonValue


PermissionDecision = Literal["allow", "deny", "ask"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]
PermissionRiskEvaluator = Callable[[dict[str, Any]], PermissionDecision]
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


class PermissionToolSpec(Protocol):
    @property
    def risk(self) -> str: ...

    @property
    def risk_evaluator(self) -> PermissionRiskEvaluator | None: ...


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
    ) -> None:
        matching_key = (tool_name, input_contains)
        self._rules = [
            r for r in self._rules if (r.tool, r.input_contains) != matching_key
        ]
        self._rules.append(PermissionRule(tool_name, decision, input_contains))

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
    ) -> PermissionPolicy:
        current = self.load()
        new_rule = PermissionRule(tool_name, decision, input_contains)
        filtered = tuple(
            r
            for r in current.rules
            if not (r.tool == tool_name and r.input_contains == input_contains)
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
    return PermissionRule(
        tool=tool,
        decision=decision,
        input_contains=input_contains if isinstance(input_contains, str) else None,
    )


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
MATCHED_RISK_EVALUATOR = "risk_evaluator"
MATCHED_HIGH_RISK = "high_risk"
MATCHED_DEFAULT = "default"

SOURCE_CONFIG = "config"
SOURCE_SESSION = "session"
SOURCE_PERSISTENT = "persistent"
SOURCE_RISK = "risk_evaluator"
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


@dataclass(frozen=True)
class PermissionEngineConfig:
    """PermissionEngine 的静态配置。"""

    static_policy: PermissionPolicy | None = None
    session_policy: SessionPermissionPolicy | None = None
    persistent_store: PersistentPermissionStore | None = None
    restricted_dirs: tuple[str, ...] = ()
    allowlist_mode: bool = False
    high_risk_requires_approval: bool = False


class PermissionEngine:
    """统一权限决策引擎。

    决策优先级（从高到低）：
    0. restricted_dirs 硬阻断
    1. 静态 deny > 执行模式 deny
    2. 静态 ask
    3. HITL 授权（session/persistent 满足前面的 ask）
    4. 静态 allow
    5. risk_evaluator（deny/ask/allow）
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

        static = self._config.static_policy
        static_decisions: list[PermissionDecision] = []
        if static is not None:
            sd = static.decide(tool_name, action_input)
            if sd is not None:
                static_decisions = [sd]

        # Tier 1a: 静态 deny
        if "deny" in static_decisions:
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=f"permission denied for tool: {tool_name}",
                matched_rule=MATCHED_STATIC_DENY,
                source=SOURCE_CONFIG,
            )

        # Tier 1b: 执行模式 deny
        if execution_decision == "deny":
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=f"tool not allowed in current execution mode: {tool_name}",
                matched_rule=MATCHED_EXECUTION_MODE,
                source=SOURCE_EXECUTION_MODE,
            )

        # Tier 2: 执行模式 ask（先检查 HITL）
        if execution_decision == "ask":
            hitl_result = self._check_hitl_grants(tool_name, action_input)
            if hitl_result is not None:
                return hitl_result
            return self._ask_approval(
                tool_name, approval_callback, tool_spec, tool_input
            )

        # Tier 3: 静态 ask（先检查 HITL）
        if "ask" in static_decisions:
            hitl_result = self._check_hitl_grants(tool_name, action_input)
            if hitl_result is not None:
                return hitl_result
            return self._ask_approval(
                tool_name, approval_callback, tool_spec, tool_input
            )

        # Tier 4: 静态 allow
        if "allow" in static_decisions:
            return PermissionEngineResult(
                decision="allow",
                blocked=False,
                matched_rule=MATCHED_STATIC_ALLOW,
                source=SOURCE_CONFIG,
            )

        # Tier 5a: 允许列表模式 — 非白名单工具 ask
        if self._config.allowlist_mode:
            hitl_result = self._check_hitl_grants(tool_name, action_input)
            if hitl_result is not None:
                return hitl_result
            return self._ask_approval(
                tool_name, approval_callback, tool_spec, tool_input
            )

        # Tier 5b: risk_evaluator 动态决策
        risk_result = self._evaluate_risk(
            tool_name,
            action_input,
            tool_spec,
            tool_input,
            approval_callback,
        )
        if risk_result is not None:
            return risk_result

        # Tier 6: 高风险工具默认审批
        if self._should_check_high_risk(tool_spec):
            hitl_result = self._check_hitl_grants(tool_name, action_input)
            if hitl_result is not None:
                return hitl_result
            return self._ask_approval(
                tool_name, approval_callback, tool_spec, tool_input
            )

        # Tier 7: 默认放行
        return PermissionEngineResult(
            decision="allow",
            blocked=False,
            matched_rule=MATCHED_DEFAULT,
            source=SOURCE_DEFAULT,
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

    def _check_hitl_grants(
        self,
        tool_name: str,
        action_input: str,
    ) -> PermissionEngineResult | None:
        session = self._config.session_policy
        if session is not None:
            sd = session.decide(tool_name, action_input)
            if sd is not None and sd != "ask":
                return PermissionEngineResult(
                    decision=sd,
                    blocked=sd == "deny",
                    matched_rule=MATCHED_SESSION_GRANT,
                    source=SOURCE_SESSION,
                    metadata=_approval_metadata(sd, "session")
                    if sd != "deny"
                    else None,
                )

        persistent = self._config.persistent_store
        if persistent is not None:
            pp = persistent.load()
            pd = pp.decide(tool_name, action_input)
            if pd is not None and pd != "ask":
                return PermissionEngineResult(
                    decision=pd,
                    blocked=pd == "deny",
                    matched_rule=MATCHED_PERSISTENT_GRANT,
                    source=SOURCE_PERSISTENT,
                    metadata=_approval_metadata(pd, "permanent")
                    if pd != "deny"
                    else None,
                )
        return None

    def _evaluate_risk(
        self,
        tool_name: str,
        action_input: str,
        tool_spec: PermissionToolSpec | None,
        tool_input: dict[str, Any] | None,
        approval_callback: PermissionApprovalCallback | None,
    ) -> PermissionEngineResult | None:
        if tool_spec is None or not tool_spec.risk_evaluator:
            return None
        risk_decision = tool_spec.risk_evaluator(tool_input or {})
        if risk_decision == "deny":
            return PermissionEngineResult(
                decision="deny",
                blocked=True,
                reason=f"permission denied for tool: {tool_name}",
                matched_rule=MATCHED_RISK_EVALUATOR,
                source=SOURCE_RISK,
            )
        if risk_decision == "ask":
            hitl_result = self._check_hitl_grants(tool_name, action_input)
            if hitl_result is not None:
                return hitl_result
            return self._ask_approval(
                tool_name, approval_callback, tool_spec, tool_input
            )
        if risk_decision == "allow":
            return PermissionEngineResult(
                decision="allow",
                blocked=False,
                matched_rule=MATCHED_RISK_EVALUATOR,
                source=SOURCE_RISK,
            )
        return None

    def _should_check_high_risk(self, tool_spec: PermissionToolSpec | None) -> bool:
        return (
            self._config.high_risk_requires_approval
            and tool_spec is not None
            and tool_spec.risk == "high"
        )

    def _ask_approval(
        self,
        tool_name: str,
        approval_callback: PermissionApprovalCallback | None,
        tool_spec: PermissionToolSpec | None,
        tool_input: dict[str, Any] | None,
    ) -> PermissionEngineResult:
        if approval_callback is not None and tool_spec is not None:
            hitl = approval_callback(tool_spec, tool_input or {})
            if hitl.decision == "deny":
                return PermissionEngineResult(
                    decision="deny",
                    blocked=True,
                    reason=f"tool {tool_name} denied by user{DENIED_BY_USER_GUIDANCE}",
                    matched_rule=MATCHED_STATIC_ASK,
                    source=SOURCE_SESSION,
                    metadata=_approval_metadata("deny", hitl.scope),
                )
            return PermissionEngineResult(
                decision="allow",
                blocked=False,
                matched_rule=MATCHED_STATIC_ASK,
                source=SOURCE_SESSION,
                metadata=_approval_metadata("allow", hitl.scope),
            )
        return PermissionEngineResult(
            decision="ask",
            blocked=True,
            reason=f"tool requires approval: {tool_name}",
            matched_rule=MATCHED_STATIC_ASK,
        )


# ── 向后兼容包装 ──

PermissionCheckResult = PermissionEngineResult


def check_tool_permission(
    tool_name: str,
    action_input: str,
    *,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: PermissionApprovalCallback | None = None,
    tool_spec: PermissionToolSpec | None = None,
    tool_input: dict[str, Any] | None = None,
    high_risk_requires_approval: bool = False,
) -> PermissionEngineResult:
    """统一权限决策入口（向后兼容包装）。

    将参数转换为 PermissionEngine 配置并委托决策。
    """
    config = PermissionEngineConfig(
        static_policy=permission_policy,
        high_risk_requires_approval=high_risk_requires_approval,
    )
    engine = PermissionEngine(config)
    return engine.decide(
        tool_name,
        action_input,
        tool_spec=tool_spec,
        tool_input=tool_input,
        approval_callback=approval_callback,
    )
