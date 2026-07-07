"""Xiaomi MiMo provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。"""

from __future__ import annotations

from typing import Any

from xcode.ai.types import ProviderConfig

from ._compat import OpenAICompatProvider

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


class MiMoProvider(OpenAICompatProvider):
    """Xiaomi MiMo Chat API 适配。"""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(config, transport="mimo_chat", client=client)

    def _record_usage(self, response, sent_messages: int) -> None:
        super()._record_usage(response, sent_messages)
        cached = self._metrics.get("cached_tokens", 0)
        if isinstance(cached, int) and cached > 0:
            self._metrics["cache_hit_tokens"] = cached
            self._metrics["cache_miss_tokens"] = self._metrics.get(
                "prompt_cache_miss_tokens", 0
            )
