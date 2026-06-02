from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from xcode.ai.events import ProviderEvent

"""Anthropic Messages API provider（占位，可直接迁入实际实现）。"""


class AnthropicProvider:
    """Anthropic Messages API 适配。"""

    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        self.api_key = api_key
        self.model = model

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[ProviderEvent]:
        # TODO: 实现真实 Anthropic Messages API 流式调用
        raise NotImplementedError("AnthropicProvider 尚未实现")
