"""权限引擎裁决语义测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from xcode.agent.types import ApprovalRequest, ToolSpec
from xcode.coding_agent.assembly.security import sensitive_path_overrides_from_security
from xcode.harness.config import SecurityRuntimeConfig
from xcode.harness.security import HITLResult
from xcode.harness.security.permissions import PermissionEngine, PermissionEngineConfig
from xcode.harness.security.permission_model import (
    Action,
    Constraint,
    Rule,
    SensitivePathOverride,
)


def _bash_tool() -> ToolSpec:
    return ToolSpec("bash", "", "", lambda _data, _update: "")


def test_build_routes_unknown_shell_command_to_approval() -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> HITLResult:
        requests.append(request)
        return HITLResult("allow", "once")

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="ask"),),
            mode_fallback="ask",
            shell_unresolved_policy="ask",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "git log --oneline -10"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
    )

    assert result.decision == "allow"
    assert result.blocked is False
    assert len(requests) == 1


def test_act_still_asks_for_unknown_shell_command() -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> HITLResult:
        requests.append(request)
        return HITLResult("allow", "once")

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="ask"),),
            mode_fallback="ask",
            shell_unresolved_policy="ask",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "pytest -q"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
    )

    assert result.decision == "allow"
    assert len(requests) == 1


def test_never_policy_rejects_ask_without_calling_reviewer() -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> HITLResult:
        requests.append(request)
        return HITLResult("allow", "once")

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="ask"),),
            mode_fallback="ask",
            approval_policy="never",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "pytest -q"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
    )

    assert result.decision == "deny"
    assert result.reason_code == "approval_policy_never"
    assert requests == []


def test_auto_review_metadata_is_preserved() -> None:
    def approve(_request: ApprovalRequest) -> HITLResult:
        return HITLResult(
            "allow",
            "once",
            rationale="Focused local test command.",
            risk="low",
            authorization="high",
        )

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="ask"),),
            mode_fallback="ask",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "pytest -q"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
        approvals_reviewer="auto_review",
    )

    assert result.decision == "allow"
    assert result.source == "auto_review"
    assert result.metadata == {
        "approval_decision": "allow",
        "approval_scope": "once",
        "approval_reviewer": "auto_review",
        "approval_review_status": "completed",
        "approval_rationale": "Focused local test command.",
        "approval_risk": "low",
        "approval_authorization": "high",
    }


def test_build_respects_explicit_user_shell_ask() -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> HITLResult:
        requests.append(request)
        return HITLResult("allow", "once")

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="allow"),),
            user_ruleset=(Rule(action="bash", effect="ask"),),
            mode_fallback="allow",
            shell_unresolved_policy="allow",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "pytest -q"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
    )

    assert result.decision == "allow"
    assert len(requests) == 1


def test_build_respects_explicit_user_shell_deny() -> None:
    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="allow"),),
            user_ruleset=(Rule(action="bash", effect="deny"),),
            mode_fallback="allow",
            shell_unresolved_policy="allow",
        )
    )

    result = engine.decide("bash", {"command": "pytest -q"})

    assert result.decision == "deny"
    assert result.blocked is True


def test_hook_constraint_uses_the_canonical_resolver() -> None:
    class DenyProvider:
        def evaluate(self, action: Action) -> tuple[Constraint, ...]:
            return (
                Constraint(
                    decision="deny",
                    source="hook",
                    reason=f"hook denied {action.tool}",
                ),
            )

    engine = PermissionEngine(
        PermissionEngineConfig(
            hook_constraint_providers=(DenyProvider(),),
            mode_fallback="allow",
        )
    )

    result = engine.decide("read_file", {"path": "README.md"})

    assert result.decision == "deny"
    assert result.source == "hook"


def test_build_keeps_dangerous_shell_command_denied() -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> HITLResult:
        requests.append(request)
        return HITLResult("allow", "once")

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="allow"),),
            mode_fallback="allow",
            shell_unresolved_policy="allow",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "rm -rf /"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
    )

    assert result.decision == "deny"
    assert result.blocked is True
    assert requests == []


def test_auto_review_cannot_create_session_grant() -> None:
    def approve(_request: ApprovalRequest) -> HITLResult:
        return HITLResult("allow", "session")

    engine = PermissionEngine(
        PermissionEngineConfig(
            mode_ruleset=(Rule(action="bash", effect="ask"),),
            mode_fallback="ask",
        )
    )

    result = engine.decide(
        "bash",
        {"command": "pytest -q"},
        tool_spec=_bash_tool(),
        approval_callback=approve,
        approvals_reviewer="auto_review",
    )

    assert result.decision == "deny"
    assert result.reason_code == "invalid_auto_review_scope"


def test_unresolved_restricted_path_is_explicit_deny() -> None:
    engine = PermissionEngine(PermissionEngineConfig(restricted_dirs=("secrets",)))

    result = engine.decide("read_file", {})

    assert result.decision == "deny"
    assert result.blocked is True
    assert result.matched_rule == "restricted_dirs"
    assert result.reason_code == "unresolved_path_with_restricted_dirs"
    assert result.overrideable is False
    assert result.remediation == (
        "Use a command or tool input with a statically resolvable path."
    )
    assert result.reason == (
        "filesystem paths could not be extracted safely while "
        "restricted_dirs is configured for tool: read_file"
    )


def test_external_path_denial_has_actionable_remediation(tmp_path: Path) -> None:
    engine = PermissionEngine(PermissionEngineConfig(project_root=tmp_path))
    outside = tmp_path.parent / "outside.txt"

    result = engine.decide("read_file", {"path": str(outside)})

    assert result.decision == "deny"
    assert result.reason_code == "outside_approved_roots"
    assert result.overrideable is False
    assert result.remediation == (
        "Add the directory to external_directories with the required access."
    )


def test_multi_target_approval_only_offers_once(tmp_path: Path) -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> HITLResult:
        requests.append(request)
        return HITLResult("allow", "once")

    engine = PermissionEngine(PermissionEngineConfig(project_root=tmp_path))
    tool = ToolSpec("bash", "", "", lambda _data, _update: "")

    result = engine.decide(
        "bash",
        {"command": "cp source.txt target.txt"},
        tool_spec=tool,
        approval_callback=approve,
    )

    assert result.decision == "allow"
    assert requests[0].allowed_scopes == ("once",)


def test_invalid_approval_scope_is_rejected_instead_of_downgraded(
    tmp_path: Path,
) -> None:
    def approve(_request: ApprovalRequest) -> HITLResult:
        return HITLResult("allow", "permanent")

    engine = PermissionEngine(PermissionEngineConfig(project_root=tmp_path))
    tool = ToolSpec("bash", "", "", lambda _data, _update: "")

    result = engine.decide(
        "bash",
        {"command": "cp source.txt target.txt"},
        tool_spec=tool,
        approval_callback=approve,
    )

    assert result.decision == "deny"
    assert result.blocked is True
    assert result.reason_code == "invalid_approval_scope"
    assert result.approval_result is None


def test_exact_environment_file_read_override_is_allowed(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    engine = PermissionEngine(
        PermissionEngineConfig(
            project_root=tmp_path,
            sensitive_path_overrides=(
                SensitivePathOverride(path=env_path, access="read"),
            ),
            mode_fallback="allow",
        )
    )

    result = engine.decide("read_file", {"path": str(env_path)})

    assert result.decision == "allow"
    assert result.blocked is False


def test_environment_override_is_exact_and_access_scoped(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    engine = PermissionEngine(
        PermissionEngineConfig(
            project_root=tmp_path,
            sensitive_path_overrides=(
                SensitivePathOverride(path=env_path, access="read"),
            ),
            mode_fallback="allow",
        )
    )

    write_result = engine.decide("write_file", {"path": str(env_path)})
    other_result = engine.decide("read_file", {"path": str(tmp_path / ".env.local")})

    assert write_result.decision == "deny"
    assert write_result.reason_code == "sensitive_path"
    assert other_result.decision == "deny"
    assert other_result.reason_code == "sensitive_path"


def test_credential_path_cannot_use_sensitive_override(tmp_path: Path) -> None:
    key_path = tmp_path / ".ssh" / "id_rsa"
    engine = PermissionEngine(
        PermissionEngineConfig(
            project_root=tmp_path,
            sensitive_path_overrides=(
                SensitivePathOverride(path=key_path, access="read"),
            ),
            mode_fallback="allow",
        )
    )

    result = engine.decide("read_file", {"path": str(key_path)})

    assert result.decision == "deny"
    assert result.reason_code == "sensitive_path"
    assert result.remediation == (
        "Use a non-sensitive file; credential paths cannot be approved or overridden."
    )


def test_sensitive_override_runtime_config_is_normalized(tmp_path: Path) -> None:
    security = SecurityRuntimeConfig.model_validate(
        {"sensitive_path_overrides": [{"path": ".env", "access": "read"}]}
    )

    overrides = sensitive_path_overrides_from_security(security, tmp_path)

    assert overrides == (SensitivePathOverride(path=tmp_path / ".env", access="read"),)


def test_sensitive_override_runtime_config_rejects_globs() -> None:
    with pytest.raises(ValidationError, match="must be an exact path"):
        SecurityRuntimeConfig.model_validate(
            {"sensitive_path_overrides": [{"path": "**/.env", "access": "read"}]}
        )
