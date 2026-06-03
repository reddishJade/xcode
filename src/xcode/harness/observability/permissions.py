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

    def __init__(self, sandbox: SettingsSandboxPermissionPolicy, inner: Any) -> None:
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


@dataclass(frozen=True)
class PermissionCheckResult:
    """权限决策结果。

    blocked=True 时 reason 描述拦截原因；
    blocked=False 时允许执行。
    """

    blocked: bool
    reason: str = ""


def check_tool_permission(
    tool_name: str,
    action_input: str,
    *,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: Any | None = None,
    tool_spec: Any | None = None,
    tool_input: dict[str, Any] | None = None,
) -> PermissionCheckResult:
    """统一权限决策入口。

    合并 PermissionPolicy.decide() 和 risk_evaluator 两层检查：
    1. PermissionPolicy 返回 "deny" → 阻断
    2. PermissionPolicy 返回 "ask" 或 risk_evaluator 返回 "ask" → 需要 approval
    3. risk_evaluator 返回 "deny" → 阻断
    4. 否则放行
    """
    if permission_policy is not None:
        decision = permission_policy.decide(tool_name, action_input)
        if decision == "deny":
            return PermissionCheckResult(
                blocked=True, reason=f"permission denied for tool: {tool_name}"
            )
        if decision == "ask":
            if approval_callback is not None and tool_spec is not None:
                hitl = approval_callback(tool_spec, tool_input or {})
                if hitl.decision == "deny":
                    return PermissionCheckResult(
                        blocked=True, reason=f"用户拒绝了 {tool_name}"
                    )
            else:
                return PermissionCheckResult(
                    blocked=True, reason=f"permission denied for tool: {tool_name}"
                )

    if tool_spec is not None and tool_spec.risk_evaluator:
        risk_decision = tool_spec.risk_evaluator(tool_input or {})
        if risk_decision == "deny":
            return PermissionCheckResult(
                blocked=True, reason=f"permission denied for tool: {tool_name}"
            )
        if risk_decision == "ask":
            if approval_callback is not None:
                hitl = approval_callback(tool_spec, tool_input or {})
                if hitl.decision == "deny":
                    return PermissionCheckResult(
                        blocked=True, reason=f"用户拒绝了 {tool_name}"
                    )

    return PermissionCheckResult(blocked=False)
