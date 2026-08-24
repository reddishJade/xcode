"""OpenAI Chat Completions provider。"""

from __future__ import annotations

import logging
from typing import Any

from xcode.ai.types import ProviderConfig, ToolDefinition

from ._compat import OpenAICompatProvider
from ._runtime import ProviderRuntime

_LOGGER = logging.getLogger(__name__)


class OpenAIChatProvider(OpenAICompatProvider):
    """OpenAI Chat Completions provider（兼容所有 OpenAI API 兼容服务）。"""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any | None = None,
        runtime: ProviderRuntime | None = None,
    ) -> None:
        super().__init__(
            config,
            transport="openai_chat",
            client=client,
            runtime=runtime,
        )

    def _build_thinking_params(
        self,
        params: dict[str, Any],
        thinking_override: bool | None = None,
    ) -> None:
        effective = (
            self.config.thinking if thinking_override is None else thinking_override
        )
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort
        elif not effective:
            params["reasoning_effort"] = "none"

    def _build_chat_params(
        self,
        api_messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> dict[str, Any]:
        params = super()._build_chat_params(api_messages, tools)
        if self.config.response_format:
            params["response_format"] = self.config.response_format
        return params

    def _warn_builtin_tools(self, tools: tuple[ToolDefinition, ...]) -> None:
        _warn_chat_builtin_tools(tools)


def _warn_chat_builtin_tools(tools: tuple[ToolDefinition, ...]) -> None:
    for tool in tools:
        if tool.builtin is None:
            continue
        builtin_type = tool.builtin.get("type")
        if not builtin_type:
            continue
        _LOGGER.warning(
            "OpenAI Chat Completions does not support builtin tool %r "
            "with type=%r; builtin tools are not available",
            tool.name,
            builtin_type,
        )
