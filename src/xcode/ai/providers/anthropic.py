"""Anthropic Messages API provider（尚未实现）。

注册在 PROVIDER_REGISTRY 中供后续实现，选择此 provider 会立即报错。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition

NOT_IMPLEMENTED_MSG = (
    "AnthropicProvider is not implemented. "
    "Select a different provider in your configuration."
)


class AnthropicProvider:
    """Anthropic Messages API 占位。选择此 provider 会立即抛出 RuntimeError。"""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        raise RuntimeError(NOT_IMPLEMENTED_MSG)
