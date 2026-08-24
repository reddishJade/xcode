"""Provider 回退包装器，按连续错误阈值切换回退 provider。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from inspect import Parameter, signature

from xcode.ai.events import Message, ProviderEvent
from xcode.ai.providers.base import ModelProvider
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.ai.usage import UsageTotals


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

    def replace_primary(self, primary: ModelProvider) -> None:
        """热替换主 provider 并重置容灾计数。

        /model、/thinking、/effort 等命令重建主 provider 后调用此方法，
        原地换主以保留 fallback 容灾包装层。新主 provider 视为全新实例，
        下一轮 stream 优先尝试新主而非沿用 fallback 状态。
        """
        self._primary = primary
        self._consecutive_errors = 0
        self._fallback_successes = 0
        self._using_fallback = False

    @property
    def active_provider(self) -> ModelProvider:
        return self._fallback if self._using_fallback else self._primary

    @property
    def model(self) -> str:
        return str(self.active_provider.model)

    @property
    def base_url(self) -> str:
        return self.active_provider.base_url

    @property
    def transport(self) -> str:
        return self.active_provider.transport

    @property
    def thinking(self) -> bool:
        return self.active_provider.thinking

    @property
    def reasoning_effort(self) -> str | None:
        return self.active_provider.reasoning_effort

    @property
    def context_window(self) -> int | None:
        return self.active_provider.context_window

    @property
    def usage_totals(self) -> UsageTotals:
        """返回主备 provider 的累计用量之和。"""
        return self._primary.usage_totals.add(self._fallback.usage_totals)

    @property
    def cache_hit_rate(self) -> float | None:
        return self.active_provider.cache_hit_rate

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
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        provider = self._fallback if self._using_fallback else self._primary
        emitted = False
        try:
            async for event in self._stream_with(
                provider, messages, tools, options, kwargs
            ):
                emitted = True
                yield event
        except Exception:
            self._record_failure(provider)
            if (
                provider is self._primary
                and self._consecutive_errors >= self._error_threshold
            ):
                self._using_fallback = True
                self._fallback_successes = 0
                if not emitted:
                    async for event in self._stream_fallback_after_failure(
                        messages, tools, options, kwargs
                    ):
                        yield event
                    return
            raise
        else:
            self._record_success(provider)

    async def _stream_fallback_after_failure(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None,
        kwargs: dict[str, object],
    ) -> AsyncIterator[ProviderEvent]:
        """主 provider 在输出前失败时，流式执行并记录 fallback 结果。"""
        try:
            async for event in self._stream_with(
                self._fallback, messages, tools, options, kwargs
            ):
                yield event
        except Exception:
            self._record_failure(self._fallback)
            raise
        else:
            self._record_success(self._fallback)

    def _record_failure(self, provider: ModelProvider) -> None:
        self._consecutive_errors += 1
        if provider is self._fallback:
            self._fallback_successes = 0

    def _record_success(self, _provider: ModelProvider) -> None:
        self._consecutive_errors = 0

    @staticmethod
    async def _stream_with(
        provider: ModelProvider,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None,
        kwargs: dict[str, object],
    ) -> AsyncIterator[ProviderEvent]:
        if _accepts_stream_options(provider):
            async for event in provider.stream(
                messages, tools, options=options, **kwargs
            ):
                yield event
            return
        async for event in provider.stream(messages, tools):
            yield event


class _FallbackWithRetryPrimary(_FallbackSwitchingProvider):
    """扩展 _FallbackSwitchingProvider，在回退成功达到阈值后重试主 provider。"""

    def _record_success(self, provider: ModelProvider) -> None:
        super()._record_success(provider)
        if provider is not self._fallback:
            self._fallback_successes = 0
            return
        self._fallback_successes += 1
        if self._fallback_successes >= self._fallback_success_threshold:
            self._using_fallback = False
            self._fallback_successes = 0


def _accepts_stream_options(provider: ModelProvider) -> bool:
    """判断 provider.stream 是否支持标准 options/关键字参数。"""
    try:
        parameters = signature(provider.stream).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == "options" or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )
