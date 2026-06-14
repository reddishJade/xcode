from __future__ import annotations

import unittest

from xcode.harness.skills import ToolSpec, build_tool_prompt
from xcode.harness.observability import HITLResult
from xcode.tests.fixtures import run_tool


class XcodeSkillCoreTests(unittest.TestCase):
    def test_high_risk_tool_requires_approval(self) -> None:
        """高风险审批已移除 (STEP 5)；工具默认 allow。"""
        from xcode.harness.observability import PermissionPolicy, PermissionRule

        tool = ToolSpec(
            name="danger",
            description="Dangerous sample tool.",
            input_hint="anything",
            handler=lambda data: f"ran {data['input']}",
        )
        registry = {tool.name: tool}

        # 无 risk 审批 → 默认 allow
        self.assertEqual(
            run_tool(registry, "danger", {"input": "x"}),
            "ran x",
        )

        # approval_callback 在静态 ask 时仍有效
        def deny_cb(_tool: object, _input: dict) -> HITLResult:
            return HITLResult("deny", "once")

        self.assertIn(
            "denied by user",
            run_tool(
                registry,
                "danger",
                {"input": "x"},
                deny_cb,
                permission_policy=PermissionPolicy((PermissionRule("danger", "ask"),)),
            ),
        )
        self.assertEqual(
            run_tool(
                registry,
                "danger",
                {"input": "x"},
                lambda _tool, _input: HITLResult("allow", "once"),
            ),
            "ran x",
        )

    def test_tool_prompt_handles_empty_registry(self) -> None:
        prompt = build_tool_prompt(())
        self.assertEqual(prompt, "(none)")


if __name__ == "__main__":
    unittest.main()
