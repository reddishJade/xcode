"""Provider 通用流选项下发测试。"""

from __future__ import annotations

import threading
from typing import Any

import pytest

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
    from xcode.ai.providers.registry import ProviderSettings, build_provider_bundle
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


def _chunk(text: str) -> Any:
    """构建最简流式 chunk。"""

    class Delta:
        content = text
        reasoning_content = None
        tool_calls = None

    class Choice:
        delta = Delta()

    class Chunk:
        def __init__(self) -> None:
            self.usage = None
            self.choices = [Choice()]

    return Chunk()


class _CloseableStream:
    """产出首个 chunk 后阻塞，close() 后使后续读取抛异常。"""

    def __init__(self, chunk: Any) -> None:
        self._chunk = chunk
        self._pending = True
        self._release = threading.Event()
        self.closed = False

    def __iter__(self) -> _CloseableStream:
        return self

    def __next__(self) -> Any:
        if self.closed:
            raise RuntimeError("stream closed")
        if self._pending:
            self._pending = False
            return self._chunk
        self._release.wait(timeout=5)
        raise StopIteration

    def close(self) -> None:
        self.closed = True
        self._release.set()


class _StreamingClient:
    def __init__(self, stream: _CloseableStream) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self._stream = stream
        self.chat.completions.create = lambda **kwargs: self._stream


async def test_abort_active_stream_closes_inflight_stream() -> None:
    """打断后 in-flight 流被关闭，阻塞读取随即失败。"""
    stream = _CloseableStream(_chunk("hi"))
    provider = OpenAIChatProvider(
        ProviderConfig(api_key="test", model="test-model"),
        client=_StreamingClient(stream),
    )

    events = provider.stream([{"role": "user", "content": "hi"}], [])
    first = await anext(events)
    assert first.chunk == "hi"

    provider.abort_active_stream()
    assert stream.closed
    with pytest.raises(RuntimeError, match="stream closed"):
        await anext(events)
