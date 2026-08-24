"""Provider fallback 的请求级状态测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from xcode.ai.events import Message, ProviderEvent, TextDelta
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime.fallback import _FallbackWithRetryPrimary


@dataclass
class _ScriptedProvider:
    name: str
    outcomes: list[list[ProviderEvent] | Exception]
    calls: int = 0
    seen_options: list[StreamOptions | None] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.name

    @property
    def base_url(self) -> str:
        return "https://example.test"

    @property
    def transport(self) -> str:
        return "test"

    @property
    def thinking(self) -> bool:
        return False

    @property
    def reasoning_effort(self) -> str | None:
        return None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        del messages, tools, kwargs
        self.calls += 1
        self.seen_options.append(options)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        for event in outcome:
            yield event


async def _collect(provider: _FallbackWithRetryPrimary) -> list[ProviderEvent]:
    return [event async for event in provider.stream([], [])]


@pytest.mark.asyncio
async def test_fallback_recovery_counts_completed_requests_not_events() -> None:
    primary = _ScriptedProvider(
        "primary",
        [RuntimeError("one"), RuntimeError("two"), RuntimeError("three"), []],
    )
    many_events = [TextDelta(chunk=str(index)) for index in range(5)]
    fallback = _ScriptedProvider(
        "fallback",
        [many_events, many_events, many_events],
    )
    provider = _FallbackWithRetryPrimary(primary, fallback)

    with pytest.raises(RuntimeError, match="one"):
        await _collect(provider)
    with pytest.raises(RuntimeError, match="two"):
        await _collect(provider)

    assert await _collect(provider) == many_events
    assert provider.active_provider is fallback
    assert await _collect(provider) == many_events
    assert provider.active_provider is fallback
    assert await _collect(provider) == many_events
    assert provider.active_provider is primary


@pytest.mark.asyncio
async def test_partial_primary_failure_does_not_duplicate_with_fallback() -> None:
    class _PartialProvider(_ScriptedProvider):
        async def stream(
            self,
            messages: list[Message],
            tools: list[ToolDefinition],
            options: StreamOptions | None = None,
            **kwargs: object,
        ) -> AsyncIterator[ProviderEvent]:
            del messages, tools, options, kwargs
            self.calls += 1
            yield TextDelta(chunk="partial")
            raise RuntimeError("stream broke")

    primary = _PartialProvider("primary", [])
    fallback = _ScriptedProvider("fallback", [[TextDelta(chunk="fallback")]])
    provider = _FallbackWithRetryPrimary(
        primary,
        fallback,
        error_threshold=1,
    )
    seen: list[ProviderEvent] = []

    with pytest.raises(RuntimeError, match="stream broke"):
        async for event in provider.stream([], []):
            seen.append(event)

    assert seen == [TextDelta(chunk="partial")]
    assert fallback.calls == 0
    assert provider.active_provider is fallback
