"""ToolGate 纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.tool_gate import _stricter_decision


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
