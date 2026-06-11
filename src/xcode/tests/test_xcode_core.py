from __future__ import annotations

import unittest

from xcode.harness.skills import ToolSpec, build_tool_prompt
from xcode.harness.observability import HITLResult
from xcode.tests.fixtures import run_tool


class XcodeSkillCoreTests(unittest.TestCase):
    def test_high_risk_tool_requires_approval(self) -> None:
        tool = ToolSpec(
            name="danger",
            description="Dangerous sample tool.",
            input_hint="anything",
            handler=lambda data: f"ran {data['input']}",
            risk="high",
        )
        registry = {tool.name: tool}

        self.assertIn("requires approval", run_tool(registry, "danger", {"input": "x"}))
        self.assertIn(
            "denied by user",
            run_tool(
                registry,
                "danger",
                {"input": "x"},
                lambda _tool, _input: HITLResult("deny", "once"),
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
