from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from xcode.ai.providers.mimo import MiMoProvider
from xcode.ai.types import StreamOptions


class XcodeMiMoStreamOptionsTests(unittest.TestCase):
    @patch("litellm.completion")
    def test_stream_options_injection_via_public_entry(self, mock_completion) -> None:
        """验证 StreamOptions 通过 provider.stream() 注入到 MiMo 请求。"""
        captured_params: dict[str, Any] = {}

        def capture_completion(**kwargs):
            captured_params.update(kwargs)
            return iter([FakeStreamChunk(content="ok")])

        mock_completion.side_effect = capture_completion

        provider = MiMoProvider(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
            model="mimo-v2.5-pro",
        )

        async def run_test():
            options = StreamOptions(
                headers={"X-Custom": "test-header"},
                session_id="test-session-123",
                api_key="override-key",
            )
            events = [
                ev
                async for ev in provider.stream(
                    [{"role": "user", "content": "Hi"}], [], options=options
                )
            ]
            return events

        events = asyncio.run(run_test())

        self.assertEqual(captured_params.get("api_key"), "override-key")
        extra_headers = captured_params.get("extra_headers", {})
        self.assertEqual(extra_headers.get("X-Custom"), "test-header")
        self.assertEqual(extra_headers.get("x-session-id"), "test-session-123")
        self.assertTrue(len(events) > 0)


class FakeStreamChunk:
    def __init__(self, content=None) -> None:
        self.choices = [FakeStreamChoice(content)] if content else []
        self.usage = None


class FakeStreamChoice:
    def __init__(self, content) -> None:
        self.delta = FakeStreamDelta(content)


class FakeStreamDelta:
    def __init__(self, content) -> None:
        self.content = content
        self.reasoning_content = None
        self.tool_calls = []


if __name__ == "__main__":
    unittest.main()
