"""OpenAI 兼容 provider 共享的指标记录逻辑。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
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
            "reasoning_tokens": 0,
        }

    def _ensure_metrics(self) -> None:
        """惰性初始化 metrics 字典。"""
        if not hasattr(self, "metrics"):
            self.metrics = self._default_metrics()

    def _intercept_usage(
        self, chunks: Iterable[Any], message_count: int
    ) -> Iterator[Any]:
        """拦截流式响应，在遇到 usage 时记录指标。"""
        for chunk in chunks:
            usage = getattr(chunk, "usage", None)
            if usage:
                self._record_usage(chunk, message_count)
            yield chunk

    def _record_usage(self, response: Any, sent_messages: int) -> None:
        """记录 usage 指标。

        基础实现处理 OpenAI Chat Completions 标准字段：
        prompt_tokens_details.cached_tokens 和
        completion_tokens_details.reasoning_tokens。

        DeepSeek 和 ChatGLM 覆写此方法以处理 provider 专属字段。
        """
        self._ensure_metrics()
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = (
                getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            )
            self.metrics["cached_tokens"] = cached or 0

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0