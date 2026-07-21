"""权限引擎裁决语义测试。"""

from __future__ import annotations

from pathlib import Path

from xcode.harness.security.permissions import PermissionEngine, PermissionEngineConfig


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
