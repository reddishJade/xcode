"""权限引擎裁决语义测试。"""

from __future__ import annotations

from pathlib import Path

from xcode.agent.types import ApprovalRequest, ToolSpec
from xcode.harness.security.permissions import (
    HITLResult,
    PermissionEngine,
    PermissionEngineConfig,
)


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
