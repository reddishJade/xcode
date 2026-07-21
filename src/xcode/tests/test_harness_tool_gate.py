"""ToolGate 纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.tool_gate import _permission_notice, _stricter_decision
from xcode.harness.security import PermissionEngineResult
from xcode.harness.security.permission_model import ApprovalResult


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
