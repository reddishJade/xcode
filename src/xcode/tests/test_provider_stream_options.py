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
