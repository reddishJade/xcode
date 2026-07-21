"""权限引擎裁决语义测试。"""

from __future__ import annotations

from xcode.harness.security.permissions import PermissionEngine, PermissionEngineConfig


def test_unresolved_restricted_path_is_explicit_deny() -> None:
    engine = PermissionEngine(
        PermissionEngineConfig(restricted_dirs=("secrets",))
    )

    result = engine.decide("read_file", {})

    assert result.decision == "deny"
    assert result.blocked is True
    assert result.matched_rule == "restricted_dirs"
    assert result.reason == (
        "filesystem paths could not be extracted safely while "
        "restricted_dirs is configured for tool: read_file"
    )
