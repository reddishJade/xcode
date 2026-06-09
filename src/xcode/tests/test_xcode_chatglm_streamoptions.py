from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from xcode.ai.providers.chatglm import ChatGLMProvider
from xcode.ai.types import StreamOptions


class XcodeChatGLMStreamOptionsTests(unittest.TestCase):
    @patch("litellm.completion")
    def test_stream_options_injection_via_public_entry(self, mock_completion) -> None:
        """验证 StreamOptions 通过 provider.stream() 注入到 ChatGLM 请求。"""
        captured_params: dict[str, Any] = {}

        def capture_completion(**kwargs):
            captured_params.update(kwargs)
            return iter([FakeStreamChunk(content="ok")])

        mock_completion.side_effect = capture_completion

        provider = ChatGLMProvider(
            api_key="glm-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4-flash",
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

    @patch("litellm.completion")
    def test_response_format_from_constructor(self, mock_completion) -> None:
        """验证构造时传递的 response_format 生效。"""
        captured_params: dict[str, Any] = {}

        def capture_completion(**kwargs):
            captured_params.update(kwargs)
            return iter([FakeStreamChunk(content="ok")])

        mock_completion.side_effect = capture_completion

        provider = ChatGLMProvider(
            api_key="glm-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4-flash",
            response_format={"type": "json_object"},
        )

        events = list(
            provider._stream_sync([{"role": "user", "content": "Hi"}], ())
        )

        self.assertEqual(
            captured_params.get("response_format"), {"type": "json_object"}
        )
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
