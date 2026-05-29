from __future__ import annotations

import unittest

from xcode.harness import agent_runtime as agent
from xcode.harness import skills
from xcode.harness.config import AgentConfig
from xcode.harness.observability import HITLResult


class XcodeSkillCoreTests(unittest.TestCase):
    def test_high_risk_tool_requires_approval(self) -> None:
        tool = skills.ToolSpec(
            name="danger",
            description="Dangerous sample tool.",
            input_hint="anything",
            handler=lambda value: f"ran {value}",
            risk="high",
        )
        registry = {tool.name: tool}

        self.assertIn("需要授权", skills.run_tool(registry, "danger", "x"))
        self.assertIn(
            "拒绝了",
            skills.run_tool(
                registry,
                "danger",
                "x",
                lambda _tool, _input: HITLResult("deny", "once"),
            ),
        )
        self.assertEqual(
            skills.run_tool(
                registry,
                "danger",
                "x",
                lambda _tool, _input: HITLResult("allow", "once"),
            ),
            "ran x",
        )

    def test_tool_prompt_handles_empty_registry(self) -> None:
        prompt = skills.build_tool_prompt(skills.BASE_REGISTRY)
        self.assertEqual(prompt, "")


class XcodeAgentCoreTests(unittest.TestCase):
    def test_agent_runs_without_rag_or_react_imports(self) -> None:
        from xcode.tests.fixtures import FakeProvider
        from xcode.harness.agent_runtime.events import FinalMessage

        provider = FakeProvider([FinalMessage("3", "end_turn")])
        runner = agent.StructuredAgent(
            provider=provider,
            registry=skills.BASE_REGISTRY,
            config=AgentConfig(max_steps=3),
        )

        result = runner.run("what is 1+2?")

        self.assertEqual(result.answer, "3")
        self.assertFalse(result.stopped_by_limit)


if __name__ == "__main__":
    unittest.main()
