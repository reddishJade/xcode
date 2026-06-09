"""OpenAI 兼容 provider 共享的指标记录逻辑。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any


class ProviderMetricsMixin:
    """为 OpenAI Chat Completions 兼容 provider 提供 metrics 初始化、
    usage 拦截和基础指标记录。

    子类只需：
    - 在 __init__ 中调用 self._ensure_metrics()
    - 如需额外 provider 专属指标，覆写 _record_usage
    """

    metrics: dict[str, object]

    def _default_metrics(self) -> dict[str, object]:
        """返回基础 metrics 字典。子类可覆写以添加额外字段。"""
        return {
            "transport": getattr(self, "transport", "unknown"),
            "sent_messages": 0,
            "cached_tokens": 0,
            "cache_hit_rate": 0.0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
        }

    def _ensure_metrics(self) -> None:
        """惰性初始化 metrics 字典。"""
        if not hasattr(self, "metrics"):
            self.metrics = self._default_metrics()

    def _intercept_stream(
        self, stream: Iterable[Any], message_count: int
    ) -> Iterator[Any]:
        """确保 metrics 已初始化，设置 sent_messages，拦截 usage 并返回流。"""
        self._ensure_metrics()
        self.metrics["sent_messages"] = message_count
        return self._intercept_usage(stream, message_count)

    def _intercept_usage(
        self, chunks: Iterable[Any], message_count: int
    ) -> Iterator[Any]:
        """拦截流式响应，在遇到 usage 时记录指标。"""
        for chunk in chunks:
            usage = getattr(chunk, "usage", None)
            if usage:
                self._record_usage(chunk, message_count)
            yield chunk

    def _intercept_responses_stream(
        self,
        events: Iterable[Any],
        message_count: int,
        on_response_completed: Callable[[Any], None] | None = None,
    ) -> Iterator[Any]:
        """拦截 Responses API 完成事件并记录 usage。"""
        self._ensure_metrics()
        self.metrics["sent_messages"] = message_count
        for raw_event in events:
            if str(getattr(raw_event, "type", "")).endswith(".completed"):
                response = getattr(raw_event, "response", None)
                if response is not None:
                    if on_response_completed is not None:
                        on_response_completed(response)
                    self._record_usage(response, message_count)
            yield raw_event

    def _record_usage(self, response: Any, sent_messages: int) -> None:
        """记录 usage 指标。

        使用统一的缓存提取逻辑处理所有 OpenAI 兼容 provider 的通用字段。
        子类可覆写以添加 provider 专属指标（如 ChatGLM 的 prompt_tokens），
        覆写时应调用 super()._record_usage() 复用通用逻辑。
        """
        from xcode.ai.cache import extract_cache_usage

        self._ensure_metrics()
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            cache_usage = extract_cache_usage(response)
            self.metrics["cached_tokens"] = cache_usage.hit_tokens
            self.metrics["cache_hit_rate"] = cache_usage.hit_rate
            self.metrics["prompt_cache_hit_tokens"] = cache_usage.hit_tokens
            self.metrics["prompt_cache_miss_tokens"] = cache_usage.miss_tokens

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0
