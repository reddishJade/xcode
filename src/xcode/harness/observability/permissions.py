from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

"""工具执行的 allow/deny/ask 权限策略与 HITL 授权模型。"""


PermissionDecision = Literal["allow", "deny", "ask"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]


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


class PermissionPolicy:
    def __init__(self, rules: tuple[PermissionRule, ...] = ()) -> None:
        self.rules = rules

    def decide(self, tool_name: str, action_input: str) -> PermissionDecision | None:
        for rule in self.rules:
            if rule.tool != tool_name and rule.tool != "*":
                continue
            if (
                rule.input_contains is not None
                and rule.input_contains not in action_input
            ):
                continue
            return rule.decision
        return None


class SessionPermissionPolicy:
    """运行时会话级权限覆写。/clear 或新 fork 时清空。"""

    def __init__(self) -> None:
        self._rules: list[PermissionRule] = []

    def grant(
        self,
        tool_name: str,
        decision: PermissionDecision,
        input_contains: str | None = None,
    ) -> None:
        self._rules.append(PermissionRule(tool_name, decision, input_contains))

    def decide(self, tool_name: str, action_input: str) -> PermissionDecision | None:
        for rule in reversed(self._rules):
            if rule.tool != tool_name and rule.tool != "*":
                continue
            if (
                rule.input_contains is not None
                and rule.input_contains not in action_input
            ):
                continue
            return rule.decision
        return None

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
            rules = []
            for r in raw:
                if not isinstance(r, dict):
                    continue
                tool = r.get("tool", "")
                decision = r.get("decision", "")
                if decision not in ("allow", "deny", "ask"):
                    continue
                rules.append(
                    PermissionRule(
                        tool=tool,
                        decision=decision,
                        input_contains=r.get("input_contains"),
                    )
                )
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
        data: list[dict[str, Any]] = []
        for r in rules:
            entry: dict[str, Any] = {"tool": r.tool, "decision": r.decision}
            if r.input_contains is not None:
                entry["input_contains"] = r.input_contains
            data.append(entry)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


class SettingsSandboxPermissionPolicy:
    """基于 settings.json 配置的安全沙箱权限审查策略。"""

    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self.settings = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def decide(self, tool_name: str, action_input: str) -> PermissionDecision | None:
        allowed_tools = self.settings.get("allowedTools")
        denied_tools = self.settings.get("deniedTools")

        # 1. 校验黑名单工具
        if isinstance(denied_tools, list) and tool_name in denied_tools:
            return "deny"

        # 2. 校验受限目录边界
        restricted_dirs = self.settings.get("restrictedDirs")
        if isinstance(restricted_dirs, list) and restricted_dirs:
            input_lower = action_input.lower()
            for r_dir in restricted_dirs:
                if str(r_dir).lower() in input_lower:
                    return "deny"

        # 3. 校验白名单工具
        if isinstance(allowed_tools, list):
            if tool_name in allowed_tools:
                return "allow"
            else:
                return "ask"

        return None


class CompositePermissionPolicy:
    """组合权限策略，优先校验 settings.json 安全沙箱，然后回退到内层策略。"""

    def __init__(
        self, sandbox: SettingsSandboxPermissionPolicy, inner: PermissionPolicy | None
    ) -> None:
        self.sandbox = sandbox
        self.inner = inner

    def decide(self, tool_name: str, action_input: str) -> PermissionDecision | None:
        sandbox_decision = self.sandbox.decide(tool_name, action_input)
        if sandbox_decision is not None:
            return sandbox_decision
        if self.inner is not None:
            return self.inner.decide(tool_name, action_input)
        return None


# ── 统一权限决策 ──

DENIED_BY_USER_GUIDANCE = (
    "; use read-only checks (e.g. git status/git diff) or request manual execution"
)


@dataclass(frozen=True)
class PermissionCheckResult:
    """权限决策结果。

    decision 取值：
    - "allow"  — 放行
    - "deny"   — 硬性拒绝（策略或 risk_evaluator 返回 deny）
    - "ask"    — 需要人工审批但无 approval_callback 可用
    """

    blocked: bool
    reason: str = ""
    decision: Literal["allow", "deny", "ask"] = "allow"
    metadata: dict[str, Any] | None = None


def _ask_or_deny(
    approval_callback: Any | None,
    tool_spec: Any | None,
    tool_input: dict[str, Any] | None,
    tool_name: str,
) -> PermissionCheckResult:
    """统一的 'ask or deny' 审批路径。

    有 callback 时提交 HITL；无 callback 时返回 decision="ask" 阻断。
    """
    if approval_callback is not None and tool_spec is not None:
        hitl = approval_callback(tool_spec, tool_input or {})
        if hitl.decision == "deny":
            return PermissionCheckResult(
                blocked=True,
                reason=f"tool {tool_name} denied by user{DENIED_BY_USER_GUIDANCE}",
                decision="deny",
                metadata={"user_decision": "deny", "approval_scope": hitl.scope},
            )
        return PermissionCheckResult(
            blocked=False,
            decision="allow",
            metadata={"user_decision": "allow", "approval_scope": hitl.scope},
        )
    return PermissionCheckResult(
        blocked=True,
        reason=f"tool requires approval: {tool_name}",
        decision="ask",
    )


def check_tool_permission(
    tool_name: str,
    action_input: str,
    *,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: Any | None = None,
    tool_spec: Any | None = None,
    tool_input: dict[str, Any] | None = None,
    high_risk_requires_approval: bool = False,
) -> PermissionCheckResult:
    """统一权限决策入口。

    合并 PermissionPolicy.decide()、risk_evaluator 以及 high-risk 默认规则：
    1. PermissionPolicy 返回 "deny" → 阻断
    2. PermissionPolicy 返回 "ask" → 需要 approval
    3. risk_evaluator 返回 "deny" → 阻断
    4. risk_evaluator 返回 "ask" → 需要 approval
    5. high_risk_requires_approval=True 且 tool.risk=="high" → 需要 approval
    6. 否则放行

    当需要 approval 但无 approval_callback 时返回 decision="ask"。
    当 approval_callback 授权通过时 metadata 包含 user_decision 和 approval_scope。
    """
    if permission_policy is not None:
        policy_decision = permission_policy.decide(tool_name, action_input)
        if policy_decision == "deny":
            return PermissionCheckResult(
                blocked=True,
                reason=f"permission denied for tool: {tool_name}",
                decision="deny",
            )
        if policy_decision == "ask":
            return _ask_or_deny(approval_callback, tool_spec, tool_input, tool_name)
        if policy_decision == "allow":
            return PermissionCheckResult(blocked=False, decision="allow")

    if tool_spec is not None and tool_spec.risk_evaluator:
        risk_decision = tool_spec.risk_evaluator(tool_input or {})
        if risk_decision == "deny":
            return PermissionCheckResult(
                blocked=True,
                reason=f"permission denied for tool: {tool_name}",
                decision="deny",
            )
        if risk_decision == "ask":
            return _ask_or_deny(approval_callback, tool_spec, tool_input, tool_name)
        if risk_decision == "allow":
            return PermissionCheckResult(blocked=False, decision="allow")

    if (
        high_risk_requires_approval
        and tool_spec is not None
        and tool_spec.risk == "high"
    ):
        return _ask_or_deny(approval_callback, tool_spec, tool_input, tool_name)

    return PermissionCheckResult(blocked=False, decision="allow")
