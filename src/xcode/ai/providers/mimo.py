from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from xcode.ai.types import ToolDefinition

from .codec import to_chat_messages, to_chat_tool
from .openai_compat import OpenAICompatProvider

"""Xiaomi MiMo provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。

API 文档：https://platform.xiaomimimo.com/
支持模型：mimo-v2.5-pro、mimo-v2.5、mimo-v2-flash 等。
"""

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


class MiMoProvider(OpenAICompatProvider):
    """Xiaomi MiMo Chat API 适配。

    使用 OpenAI 兼容接口，支持 thinking 模式和 reasoning_content。
    MiMo 建议保留所有历史 reasoning_content 以获得最佳表现。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = MIMO_BASE_URL,
        model: str = "mimo-v2.5-pro",
        thinking: bool = True,
        runtime=None,
        client=None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            thinking=thinking,
            runtime=runtime,
            client=client,
            transport="mimo_chat",
            import_error_msg="Missing dependency: openai.",
        )

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **_kwargs: Any,
    ) -> Iterator[Any]:
        openai_messages = to_chat_messages(messages)

        params: dict[str, object] = {
            "model": self.model,
            "messages": openai_messages,
            "tools": [to_chat_tool(t.name, t.description, t.schema) for t in tools],
            "stream": True,
        }

        self._build_thinking_params(params)

        yield from self._call_chat_api(params, len(openai_messages))

    def _record_usage(self, response, sent_messages: int) -> None:
        """记录 MiMo usage，提取缓存和 reasoning 统计。"""
        from xcode.ai.cache import extract_cache_usage

        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            # 使用统一的缓存提取逻辑
            cache_usage = extract_cache_usage(response)
            self.metrics["cached_tokens"] = cache_usage.hit_tokens
            self.metrics["cache_hit_rate"] = cache_usage.hit_rate
            if cache_usage.hit_tokens > 0:
                self.metrics["cache_hit_tokens"] = cache_usage.hit_tokens
                self.metrics["cache_miss_tokens"] = cache_usage.miss_tokens

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0
