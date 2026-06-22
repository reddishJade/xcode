from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from xcode.ai.providers.chatglm import ChatGLMProvider
from xcode.ai.types import StreamOptions
import pytest


def _make_mock_client(chunks: list | None = None) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks or [])
    return client


class XcodeChatGLMStreamOptionsTests:
    def test_stream_options_injection_via_public_entry(self) -> None:
        """验证 StreamOptions 通过 provider.stream() 注入到 ChatGLM 请求。"""
        client = _make_mock_client([FakeStreamChunk(content="ok")])

        provider = ChatGLMProvider(
            api_key="glm-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4-flash",
            client=client,
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
        kwargs = client.chat.completions.create.call_args.kwargs

        assert kwargs.get("api_key") == "override-key"
        extra_headers = kwargs.get("extra_headers", {})
        assert extra_headers.get("X-Custom") == "test-header"
        assert extra_headers.get("x-session-id") == "test-session-123"
        assert len(events) > 0

    def test_response_format_from_constructor(self) -> None:
        """验证构造时传递的 response_format 生效。"""
        client = _make_mock_client([FakeStreamChunk(content="ok")])

        provider = ChatGLMProvider(
            api_key="glm-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4-flash",
            response_format={"type": "json_object"},
            client=client,
        )

        events = list(provider._stream_sync([{"role": "user", "content": "Hi"}], ()))

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs.get("response_format") == {"type": "json_object"}
        assert len(events) > 0


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
    pytest.main()
