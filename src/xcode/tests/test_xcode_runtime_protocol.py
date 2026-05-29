from __future__ import annotations

import unittest
from typing import get_type_hints

from xcode.harness.agent_runtime import events
from xcode.harness.agent_runtime.provider import ModelProvider


class RuntimeProtocolTest(unittest.TestCase):
    def test_provider_events_are_runtime_owned(self) -> None:
        names = {
            "TextDelta",
            "ToolCallReady",
            "UsageUpdate",
            "FinalMessage",
            "ToolCall",
            "ToolResult",
        }
        for name in names:
            self.assertTrue(hasattr(events, name), name)
        self.assertFalse(hasattr(events, "ToolResultsReady"))

    def test_provider_protocol_is_stream(self) -> None:
        hints = get_type_hints(ModelProvider.stream)
        ret = hints.get("return")
        assert ret is not None
        self.assertEqual(ret.__name__, "AsyncIterator")


if __name__ == "__main__":
    unittest.main()
