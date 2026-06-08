from __future__ import annotations

import asyncio
from dataclasses import dataclass
import unittest
from typing import Any

from xcode.ai.providers.factory import _build_llm_profile
from xcode.ai.providers.openai import OpenAIChatProvider, OpenAIResponsesProvider
from xcode.ai.providers.runtime import ProviderRuntime
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

    def test_chat_provider_sends_configured_response_format(self) -> None:
        """OpenAI Chat 使用 Chat Completions 的 response_format 字段。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            response_format={"type": "json_object"},
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "json"}], ()))

        self.assertEqual(
            client.chat.completions.kwargs["response_format"],
            {"type": "json_object"},
        )

    def test_responses_provider_maps_response_format_to_text_config(self) -> None:
        """Responses 使用 text.format 承载结构化输出配置。"""
        client = FakeOpenAIClient()
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    "strict": True,
                },
            },
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "json"}], ()))

        text_config = client.responses.kwargs["text"]
        self.assertEqual(text_config["format"]["type"], "json_schema")
        self.assertEqual(text_config["format"]["name"], "answer")
        self.assertTrue(text_config["format"]["strict"])

    def test_factory_preserves_openai_response_format(self) -> None:
        """factory 构建 OpenAI provider 时保留结构化输出配置。"""
        provider = _build_llm_profile(
            MockProfile(transport="openai_responses"),
            profile_name="main",
            env_files=(),
            runtime=ProviderRuntime(),
        )

        self.assertIsInstance(provider, OpenAIResponsesProvider)
        assert isinstance(provider, OpenAIResponsesProvider)
        self.assertEqual(provider.response_format, {"type": "json_object"})

    def test_responses_provider_applies_extended_response_options(self) -> None:
        """Responses 透传官方请求级控制参数。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                response_format={"type": "json_object"},
                client=client,
            )
            options = StreamOptions(
                background=True,
                include=["reasoning.encrypted_content"],
                instructions="Answer briefly.",
                max_tool_calls=2,
                parallel_tool_calls=False,
                prompt_cache_retention="24h",
                safety_identifier="user-1",
                service_tier="flex",
                store=False,
                tool_choice="auto",
                top_logprobs=3,
                top_p=0.8,
                truncation="auto",
                user="end-user",
                verbosity="low",
                response_extra_params={"custom_beta": "value", "store": True},
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}], [], options=options
                )
            ]

            kwargs = client.responses.kwargs
            self.assertIs(kwargs["background"], True)
            self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
            self.assertEqual(kwargs["instructions"], "Answer briefly.")
            self.assertEqual(kwargs["max_tool_calls"], 2)
            self.assertIs(kwargs["parallel_tool_calls"], False)
            self.assertEqual(kwargs["prompt_cache_retention"], "24h")
            self.assertEqual(kwargs["safety_identifier"], "user-1")
            self.assertEqual(kwargs["service_tier"], "flex")
            self.assertIs(kwargs["store"], False)
            self.assertEqual(kwargs["tool_choice"], "auto")
            self.assertEqual(kwargs["top_logprobs"], 3)
            self.assertEqual(kwargs["top_p"], 0.8)
            self.assertEqual(kwargs["truncation"], "auto")
            self.assertEqual(kwargs["user"], "end-user")
            self.assertEqual(kwargs["text"]["format"], {"type": "json_object"})
            self.assertEqual(kwargs["text"]["verbosity"], "low")
            self.assertEqual(kwargs["custom_beta"], "value")

        asyncio.run(run_test())


@dataclass(frozen=True)
class MockProfile:
    """模拟 provider profile 配置。"""

    transport: str
    chat_model: str = "gpt-5.4"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "test-key"
    thinking: bool = True
    reasoning_effort: str | None = "high"
    clear_thinking: bool = False
    tool_stream: bool = True
    response_format: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """为测试 profile 设置默认结构化输出配置。"""
        if self.response_format is not None:
            return
        object.__setattr__(self, "response_format", {"type": "json_object"})


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
