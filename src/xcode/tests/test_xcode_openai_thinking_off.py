"""回归测试：OpenAI provider 在 thinking=False 时发送 reasoning_effort=none。

thinking=False 优先于配置的 reasoning_effort，确保 /thinking off
在请求层生效，不被 profile 的 reasoning_effort 覆盖。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xcode.ai.providers.openai import OpenAIChatProvider
from xcode.ai.providers.runtime import ProviderRuntime
from xcode.tests.test_xcode_providers import FakeStreamChunk, _make_mock_client


def _build_provider(
    *, thinking: bool, reasoning_effort: str | None, client: MagicMock
) -> OpenAIChatProvider:
    return OpenAIChatProvider(
        api_key="test-key",
        base_url="https://api.openai.test/v1",
        model="model",
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        runtime=ProviderRuntime(),
        client=client,
    )


def _sent_kwargs(provider: OpenAIChatProvider) -> dict[str, object]:
    events = list(
        provider._stream_sync(
            [{"role": "user", "content": "hi"}],
            (),
        )
    )
    _ = events
    return provider.client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]


def test_thinking_off_with_configured_effort_sends_none() -> None:
    """thinking=False 且 reasoning_effort="high" 时必须发送 reasoning_effort=none。"""
    client = _make_mock_client([FakeStreamChunk(content="done")])
    provider = _build_provider(thinking=False, reasoning_effort="high", client=client)

    kwargs = _sent_kwargs(provider)
    assert kwargs["reasoning_effort"] == "none", (
        "thinking=False 被 reasoning_effort=high 覆盖，/thinking off 失效"
    )


def test_thinking_on_with_configured_effort_sends_effort() -> None:
    """thinking=True 时发送配置的 reasoning_effort。"""
    client = _make_mock_client([FakeStreamChunk(content="done")])
    provider = _build_provider(thinking=True, reasoning_effort="high", client=client)

    kwargs = _sent_kwargs(provider)
    assert kwargs["reasoning_effort"] == "high"


def test_thinking_off_without_effort_sends_none() -> None:
    """thinking=False 且无 reasoning_effort 时发送 none。"""
    client = _make_mock_client([FakeStreamChunk(content="done")])
    provider = _build_provider(thinking=False, reasoning_effort=None, client=client)

    kwargs = _sent_kwargs(provider)
    assert kwargs["reasoning_effort"] == "none"


def test_thinking_on_without_effort_omits_param() -> None:
    """thinking=True 且无 reasoning_effort 时不发送 reasoning_effort。"""
    client = _make_mock_client([FakeStreamChunk(content="done")])
    provider = _build_provider(thinking=True, reasoning_effort=None, client=client)

    kwargs = _sent_kwargs(provider)
    assert "reasoning_effort" not in kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
