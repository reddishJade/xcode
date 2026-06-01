from __future__ import annotations

import unittest

from xcode.harness.config import AgentConfig
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent
from xcode.experimental.speculation import SpeculationPlanner


from xcode.tests.fixtures import FakeProvider
from xcode.harness.agent_runtime.events import (
    TextDelta,
    FinalMessage,
    ToolCallReady,
    ToolCall,
)


class XcodeSpeculationTests(unittest.TestCase):
    def test_planner_prepares_diff_after_edit(self) -> None:
        event = SpeculationPlanner().plan("edit_file", "ok")

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "prepare_diff_view")

    def test_structured_agent_emits_speculation_event(self) -> None:
        from xcode.harness.agent_runtime.events import ProviderEvent

        mock_events: list[list[ProviderEvent]] = [
            [
                ToolCallReady([ToolCall("e", "edit_file", {})]),
                FinalMessage("", "end_turn"),
            ],
            [TextDelta("done"), FinalMessage("", "end_turn")],
        ]
        provider = FakeProvider(mock_events)
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec("edit_file", "Edit.", "empty", lambda _value: "edited"),
            ),
            config=AgentConfig(max_steps=2),
            speculation_planner=SpeculationPlanner(),
        )

        events = list(agent.run_stream("edit"))

        speculation = [event for event in events if event.type == "speculation"]
        self.assertEqual(speculation[0].data.kind, "prepare_diff_view")

    def test_planner_prepares_recovery_after_error(self) -> None:
        event = SpeculationPlanner().plan("bash", "error")

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.kind, "prepare_recovery_hint")


if __name__ == "__main__":
    unittest.main()
