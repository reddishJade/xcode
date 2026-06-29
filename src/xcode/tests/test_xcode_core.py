from __future__ import annotations

from xcode.harness.skills import ToolSpec, build_tool_prompt
from xcode.harness.observability import (
    HITLResult,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
)
import pytest


class XcodeSkillCoreTests:
    def test_high_risk_tool_requires_approval(self) -> None:
        """高风险审批已移除 (STEP 5)；工具默认 allow。"""
        tool = ToolSpec(
            name="danger",
            description="Dangerous sample tool.",
            input_hint="anything",
            handler=lambda data: f"ran {data['input']}",
        )

        # 默认 allow
        engine = PermissionEngine(PermissionEngineConfig())
        result = engine.decide("danger", {"input": "x"}, tool_spec=tool)
        assert not (result.blocked)
        assert tool.handler({"input": "x"}) == "ran x"

        # 静态 ask + deny callback
        def deny_cb(_tool: object, _input: dict) -> HITLResult:
            return HITLResult("deny", "once")

        engine2 = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="danger", decision="ask"),)
                ),
            )
        )
        result2 = engine2.decide(
            "danger",
            {"input": "x"},
            tool_spec=tool,
            approval_callback=deny_cb,
        )
        assert result2.blocked
        assert "denied by user" in str(result2.reason)

        # 静态 ask + allow callback
        engine3 = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="danger", decision="ask"),)
                ),
            )
        )
        result3 = engine3.decide(
            "danger",
            {"input": "x"},
            tool_spec=tool,
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
        )
        assert not (result3.blocked)
        assert tool.handler({"input": "x"}) == "ran x"

    def test_tool_prompt_handles_empty_registry(self) -> None:
        prompt = build_tool_prompt(())
        assert prompt == "(none)"


if __name__ == "__main__":
    pytest.main()
