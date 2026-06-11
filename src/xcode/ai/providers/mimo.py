from __future__ import annotations

from typing import Any

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
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            thinking=thinking,
            runtime=runtime,
            transport="mimo_chat",
            client=client,
        )

    def _record_usage(self, response, sent_messages: int) -> None:
        """记录 MiMo usage，复用基类通用逻辑后补充 MiMo 专属字段。"""
        super()._record_usage(response, sent_messages)
        cached = self.metrics.get("cached_tokens", 0)
        if isinstance(cached, int) and cached > 0:
            self.metrics["cache_hit_tokens"] = cached
            self.metrics["cache_miss_tokens"] = self.metrics.get(
                "prompt_cache_miss_tokens", 0
            )
