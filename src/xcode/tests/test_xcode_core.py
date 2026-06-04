from __future__ import annotations

import unittest

from xcode.harness import skills
from xcode.harness.observability import HITLResult


class XcodeSkillCoreTests(unittest.TestCase):
    def test_high_risk_tool_requires_approval(self) -> None:
        tool = skills.ToolSpec(
            name="danger",
            description="Dangerous sample tool.",
            input_hint="anything",
            handler=lambda data: f"ran {data['input']}",
            risk="high",
        )
        registry = {tool.name: tool}

        self.assertIn(
            "requires approval", skills.run_tool(registry, "danger", {"input": "x"})
        )
        self.assertIn(
            "denied by user",
            skills.run_tool(
                registry,
                "danger",
                {"input": "x"},
                lambda _tool, _input: HITLResult("deny", "once"),
            ),
        )
        self.assertEqual(
            skills.run_tool(
                registry,
                "danger",
                {"input": "x"},
                lambda _tool, _input: HITLResult("allow", "once"),
            ),
            "ran x",
        )

    def test_tool_prompt_handles_empty_registry(self) -> None:
        prompt = skills.build_tool_prompt(skills.BASE_REGISTRY)
        self.assertEqual(prompt, "")


if __name__ == "__main__":
    unittest.main()
