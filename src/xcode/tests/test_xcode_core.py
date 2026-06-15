from __future__ import annotations

import unittest

from xcode.harness.skills import ToolSpec, build_tool_prompt
from xcode.harness.observability import (
    HITLResult,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    PermissionRule,
)


class XcodeSkillCoreTests(unittest.TestCase):
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
        result = engine.decide(
            "danger", '{"input": "x"}', tool_spec=tool, tool_input={"input": "x"}
        )
        self.assertFalse(result.blocked)
        self.assertEqual(tool.handler({"input": "x"}), "ran x")

        # 静态 ask + deny callback
        def deny_cb(_tool: object, _input: dict) -> HITLResult:
            return HITLResult("deny", "once")

        engine2 = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("danger", "ask"),)),
            )
        )
        result2 = engine2.decide(
            "danger",
            '{"input": "x"}',
            tool_spec=tool,
            tool_input={"input": "x"},
            approval_callback=deny_cb,
        )
        self.assertTrue(result2.blocked)
        self.assertIn("denied by user", str(result2.reason))

        # 静态 ask + allow callback
        engine3 = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((PermissionRule("danger", "ask"),)),
            )
        )
        result3 = engine3.decide(
            "danger",
            '{"input": "x"}',
            tool_spec=tool,
            tool_input={"input": "x"},
            approval_callback=lambda _t, _i: HITLResult("allow", "once"),
        )
        self.assertFalse(result3.blocked)
        self.assertEqual(tool.handler({"input": "x"}), "ran x")

    def test_tool_prompt_handles_empty_registry(self) -> None:
        prompt = build_tool_prompt(())
        self.assertEqual(prompt, "(none)")


if __name__ == "__main__":
    unittest.main()
