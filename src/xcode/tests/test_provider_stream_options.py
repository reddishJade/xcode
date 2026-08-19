"""Provider 通用流选项下发测试。"""

from __future__ import annotations

from typing import Any

from xcode.ai.providers.openai import OpenAIChatProvider
from xcode.ai.types import ProviderConfig, StreamOptions


class _Completions:
    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> list[Any]:
        self.params = kwargs
        return []


class _Client:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions()


async def test_common_stream_options_reach_chat_request() -> None:
    client = _Client()
    provider = OpenAIChatProvider(
        ProviderConfig(api_key="test", model="test-model"),
        client=client,
    )

    events = provider.stream(
        [{"role": "user", "content": "hello"}],
        [],
        options=StreamOptions(
            temperature=0,
            max_tokens=123,
            top_p=0.8,
            tool_choice="auto",
            response_extra_params={"seed": 7},
        ),
    )
    assert [event async for event in events] == []

    params = client.chat.completions.params
    assert params["temperature"] == 0
    assert params["max_tokens"] == 123
    assert params["top_p"] == 0.8
    assert params["tool_choice"] == "auto"
    assert params["seed"] == 7


async def test_provider_exposes_context_window_override() -> None:
    provider = OpenAIChatProvider(
        ProviderConfig(
            api_key="test",
            model="test-model",
            context_window=262_144,
        ),
        client=_Client(),
    )
    assert provider.context_window == 262_144


async def test_provider_context_window_defaults_to_none() -> None:
    provider = OpenAIChatProvider(
        ProviderConfig(api_key="test", model="test-model"),
        client=_Client(),
    )
    assert provider.context_window is None


def test_build_provider_bundle_carries_context_window() -> None:
    from xcode.ai.providers.registry import build_provider_bundle, ProviderSettings
    from xcode.harness.config import ModelProfileRuntimeConfig

    bundle = build_provider_bundle(
        ProviderSettings(
            env_files=(),
            model_profiles={
                "main": ModelProfileRuntimeConfig(
                    api_key="test",
                    context_window=262_144,
                )
            },
        )
    )
    assert bundle.llm.context_window == 262_144
