from __future__ import annotations

import unittest
from typing import Any

from xcode.ai.providers.openai import OpenAIChatProvider


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


class FakeOpenAIClient:
    """记录 Chat Completions 请求参数的测试客户端。"""

    def __init__(self) -> None:
        self.chat = FakeChat()


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


if __name__ == "__main__":
    unittest.main()
