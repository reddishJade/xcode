from __future__ import annotations

from typing import get_type_hints

from xcode.ai import events
from xcode.ai.providers.protocol import ModelProvider
import pytest


class RuntimeProtocolTest:
    def test_provider_events_are_ai_owned(self) -> None:
        names = {
            "TextDelta",
            "ToolCallEvent",
            "UsageUpdate",
            "FinalMessage",
            "ToolCall",
        }
        for name in names:
            assert hasattr(events, name)
        assert not (hasattr(events, "ToolResult"))

    def test_provider_protocol_is_stream(self) -> None:
        hints = get_type_hints(ModelProvider.stream)
        ret = hints.get("return")
        assert ret is not None
        assert ret.__name__ == "AsyncIterator"


if __name__ == "__main__":
    pytest.main()
