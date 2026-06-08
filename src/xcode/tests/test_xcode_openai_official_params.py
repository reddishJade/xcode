from __future__ import annotations

import unittest
from typing import Any

import asyncio

from xcode.ai.providers.openai import OpenAIChatProvider, OpenAIResponsesProvider
from xcode.ai.types import StreamOptions


class XcodeOpenAIOfficialParamsTests(unittest.TestCase):
    """OpenAI 官方 API 参数边界测试。"""

    def test_chat_provider_does_not_send_provider_specific_thinking_body(self) -> None:
        """OpenAI Chat 不发送兼容 provider 专属 extra_body.thinking。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            thinking=True,
            reasoning_effort="high",
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))

        kwargs = client.chat.completions.kwargs
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_chat_provider_maps_disabled_thinking_to_none_effort(self) -> None:
        """thinking 关闭时使用官方 reasoning_effort=none。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            thinking=False,
            reasoning_effort=None,
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))

        kwargs = client.chat.completions.kwargs
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "none")

    def test_responses_provider_applies_stream_options(self) -> None:
        """Responses 公共入口透传请求级选项。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )
            options = StreamOptions(
                api_key="override-key",
                headers={"x-extra": "value"},
                session_id="session-1",
                metadata={"task": "unit"},
                max_tokens=123,
                temperature=0.2,
                timeout_ms=2500,
            )

            events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}], [], options=options
                )
            ]

            kwargs = client.responses.kwargs
            self.assertEqual(events, [])
            self.assertEqual(client.override_api_key, "override-key")
            self.assertEqual(kwargs["extra_headers"]["x-session-id"], "session-1")
            self.assertEqual(kwargs["extra_headers"]["x-extra"], "value")
            self.assertEqual(kwargs["metadata"], {"task": "unit"})
            self.assertEqual(kwargs["max_output_tokens"], 123)
            self.assertEqual(kwargs["temperature"], 0.2)
            self.assertEqual(kwargs["timeout"], 2.5)

        asyncio.run(run_test())


class FakeOpenAIClient:
    """记录 Chat Completions 请求参数的测试客户端。"""

    def __init__(self) -> None:
        self.chat = FakeChat()
        self.responses = FakeResponses()
        self.override_api_key: str | None = None

    def with_options(self, *, api_key: str) -> FakeOpenAIClient:
        """记录请求级 API key 并返回同一个测试客户端。"""
        self.override_api_key = api_key
        return self


class FakeChat:
    """模拟 OpenAI chat 命名空间。"""

    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeCompletions:
    """记录 create 调用参数。"""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        """返回空流并保存请求。"""
        self.kwargs = kwargs
        return iter([])


class FakeResponses:
    """记录 Responses create 调用参数。"""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        """返回空流并保存请求。"""
        self.kwargs = kwargs
        return iter([])


if __name__ == "__main__":
    unittest.main()
