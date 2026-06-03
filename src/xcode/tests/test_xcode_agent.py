from __future__ import annotations

import unittest

from xcode.agent.provider_response import provider_events_to_response
from xcode.agent.types import TextContent
from xcode.ai.events import (
    FinalMessage,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)


class ProviderResponseTests(unittest.TestCase):
    def test_provider_events_to_response_keeps_core_stream_semantics(self) -> None:
        response = provider_events_to_response(
            [
                ReasoningDelta("why"),
                TextDelta("hel"),
                TextDelta("lo"),
                ToolCallEvent([ToolCall("call-1", "echo", {"text": "hello"})]),
                FinalMessage("", stop_reason="tool_use"),
            ]
        )

        self.assertEqual(response.reasoning_content, "why")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(response.deltas[0].kind, "reasoning")
        self.assertEqual(response.deltas[1].chunk, "hel")
        self.assertEqual(response.content[0], TextContent(text="hello"))


if __name__ == "__main__":
    unittest.main()
