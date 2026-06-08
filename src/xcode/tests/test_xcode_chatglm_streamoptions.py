from __future__ import annotations

import asyncio
import unittest
from typing import Any

from xcode.ai.providers.chatglm import ChatGLMProvider
from xcode.ai.types import StreamOptions


class XcodeChatGLMStreamOptionsTests(unittest.TestCase):
    def test_stream_options_injection_via_public_entry(self) -> None:
        """验证 StreamOptions 通过 provider.stream() 注入到 ChatGLM 请求。"""

        async def run_test():
            captured_params: dict[str, Any] = {}

            def capture_create(**kwargs):
                captured_params.update(kwargs)
                return iter([FakeStreamChunk(content="ok")])

            client = FakeOpenAIClient(stream_chunks=[])
            client.chat.completions.create = capture_create
            provider = ChatGLMProvider(
                api_key="glm-key",
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                model="glm-4-flash",
                client=client,
            )

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

            self.assertEqual(captured_params.get("api_key"), "override-key")
            extra_headers = captured_params.get("extra_headers", {})
            self.assertEqual(extra_headers.get("X-Custom"), "test-header")
            self.assertEqual(extra_headers.get("x-session-id"), "test-session-123")
            self.assertTrue(len(events) > 0)

        asyncio.run(run_test())

    def test_response_format_from_constructor(self) -> None:
        """验证构造时传递的 response_format 生效。"""
        captured_params: dict[str, Any] = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return iter([FakeStreamChunk(content="ok")])

        client = FakeOpenAIClient(stream_chunks=[])
        client.chat.completions.create = capture_create
        provider = ChatGLMProvider(
            api_key="glm-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4-flash",
            response_format={"type": "json_object"},
            client=client,
        )

        events = list(
            provider._stream_sync([{"role": "user", "content": "Hi"}], ())
        )

        self.assertEqual(
            captured_params.get("response_format"), {"type": "json_object"}
        )
        self.assertTrue(len(events) > 0)


class FakeOpenAIClient:
    def __init__(self, stream_chunks=None) -> None:
        self.chat = FakeChat(stream_chunks)


class FakeChat:
    def __init__(self, stream_chunks) -> None:
        self.completions = FakeCompletions(stream_chunks)


class FakeCompletions:
    def __init__(self, stream_chunks) -> None:
        self.stream_chunks = stream_chunks

    def create(self, **kwargs):
        return iter(self.stream_chunks or [])


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
