"""ToolGate 纯函数单元测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from xcode.harness.agent_runtime.config import AgentRuntimeConfig, GateConfig
from xcode.harness.agent_runtime.harness import AgentHarness
from xcode.harness.agent_runtime.tool_gate import _permission_notice, _stricter_decision
from xcode.harness.security import PermissionEngineResult
from xcode.harness.security.permission_model import ApprovalResult, ExternalDirectory


class TestStricterDecision:
    def test_stricter_wins(self) -> None:
        assert _stricter_decision("allow", "deny") == "deny"
        assert _stricter_decision("ask", "deny") == "deny"

    def test_current_is_stricter(self) -> None:
        assert _stricter_decision("deny", "allow") == "deny"
        assert _stricter_decision("deny", "ask") == "deny"

    def test_same_level(self) -> None:
        assert _stricter_decision("allow", "allow") == "allow"
        assert _stricter_decision("ask", "ask") == "ask"


def test_permission_notice_describes_automatic_session_grant() -> None:
    result = PermissionEngineResult(
        decision="allow",
        blocked=False,
        matched_rule="session_grant",
        approval_result=ApprovalResult(
            decision="allow",
            scope="session",
            grant_id="grant-1",
        ),
    )

    assert _permission_notice(result) == "Allowed by session grant"


def test_agent_harness_propagates_external_directories(tmp_path: Path) -> None:
    external = ExternalDirectory(path=tmp_path / "shared", access="read")

    harness = AgentHarness(
        provider=cast(Any, object()),
        registry=(),
        gate=GateConfig(external_directories=(external,)),
        runtime=AgentRuntimeConfig(project_root=tmp_path),
    )

    assert harness.external_directories == (external,)
    assert harness._gate.snapshot().external_directories == (external,)
