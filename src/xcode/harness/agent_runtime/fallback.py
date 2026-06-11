"""Provider 回退包装器，按连续错误阈值切换回退 provider。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import StreamOptions, ToolDefinition


class _FallbackSwitchingProvider:
    """连续错误达到阈值后切换到回退 provider 的包装器。

    追踪连续错误计数，到达阈值后切换到 fallback_provider。
    在回退 provider 上连续成功达到阈值后重新尝试主 provider。
    """

    def __init__(
        self,
        primary: ModelProvider,
        fallback: ModelProvider,
        error_threshold: int = 3,
        fallback_success_threshold: int = 3,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._error_threshold = error_threshold
        self._fallback_success_threshold = fallback_success_threshold
        self._consecutive_errors: int = 0
        self._fallback_successes: int = 0
        self._using_fallback: bool = False

    @property
    def active_provider(self) -> ModelProvider:
        return self._fallback if self._using_fallback else self._primary

    @property
    def model(self) -> str:
        return getattr(self.active_provider, "model", "unknown")

    @property
    def base_url(self) -> str:
        return getattr(self.active_provider, "base_url", "")

    @property
    def transport(self) -> str:
        return getattr(self.active_provider, "transport", "")

    @property
    def thinking(self) -> bool:
        return getattr(self.active_provider, "thinking", True)

    @property
    def reasoning_effort(self) -> str | None:
        return getattr(self.active_provider, "reasoning_effort", None)

    def reset_conversation_state(self) -> None:
        """清理主备 provider 的服务端会话状态。"""
        for provider in (self._primary, self._fallback):
            reset = getattr(provider, "reset_conversation_state", None)
            if callable(reset):
                reset()
        self._consecutive_errors = 0
        self._fallback_successes = 0
        self._using_fallback = False

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        provider = self._fallback if self._using_fallback else self._primary
        try:
            async for event in self._stream_with(
                provider, messages, tools, options, kwargs
            ):
                self._consecutive_errors = 0
                yield event
        except Exception:
            self._consecutive_errors += 1
            if (
                not self._using_fallback
                and self._consecutive_errors >= self._error_threshold
                and self._fallback is not None
            ):
                self._using_fallback = True
                self._fallback_successes = 0
                async for event in self._stream_with(
                    self._fallback, messages, tools, options, kwargs
                ):
                    yield event
            else:
                raise

    @staticmethod
    async def _stream_with(
        provider: Any,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[ProviderEvent]:
        try:
            async for event in provider.stream(
                messages, tools, options=options, **kwargs
            ):
                yield event
        except TypeError:
            async for event in provider.stream(messages, tools):
                yield event


class _FallbackWithRetryPrimary(_FallbackSwitchingProvider):
    """扩展 _FallbackSwitchingProvider，在回退成功达到阈值后重试主 provider。"""

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        provider = self._fallback if self._using_fallback else self._primary
        try:
            async for event in self._stream_with(
                provider, messages, tools, options, kwargs
            ):
                if self._using_fallback:
                    self._fallback_successes += 1
                    if self._fallback_successes >= self._fallback_success_threshold:
                        self._using_fallback = False
                        self._fallback_successes = 0
                self._consecutive_errors = 0
                yield event
        except Exception:
            self._consecutive_errors += 1
            if (
                not self._using_fallback
                and self._consecutive_errors >= self._error_threshold
                and self._fallback is not None
            ):
                self._using_fallback = True
                self._fallback_successes = 0
                async for event in self._stream_with(
                    self._fallback, messages, tools, options, kwargs
                ):
                    yield event
            else:
                raise
