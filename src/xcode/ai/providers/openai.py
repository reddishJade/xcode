"""OpenAI Chat Completions provider。"""

from __future__ import annotations

import logging
from typing import Any

from xcode.ai.types import ToolDefinition

from .openai_compat import OpenAICompatProvider
from .runtime import ProviderRuntime

_LOGGER = logging.getLogger(__name__)


class OpenAIChatProvider(OpenAICompatProvider):
    """OpenAI Chat Completions provider（兼容所有 OpenAI API 兼容服务）。

    只发送 OpenAI Chat Completions 标准参数。
    DeepSeek/ChatGLM/MiMo 等专有扩展字段（如 extra_body.thinking）
    由各自的 Provider 实现，不在此处处理。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        thinking: bool = True,
        reasoning_effort: str | None = None,
        runtime: ProviderRuntime | None = None,
        response_format: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            runtime=runtime,
            transport="openai_chat",
            client=client,
        )
        self.response_format = response_format

    def _build_chat_params(
        self,
        api_messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **kwargs: Any,
    ) -> dict[str, Any]:
        params = super()._build_chat_params(api_messages, tools)
        effective_format = kwargs.get("response_format") or self.response_format
        if effective_format:
            params["response_format"] = effective_format
        self._build_openai_reasoning_params(params)
        return params

    def _warn_builtin_tools(self, tools: tuple[ToolDefinition, ...]) -> None:
        _warn_chat_builtin_tools(tools)

    def _build_openai_reasoning_params(self, params: dict[str, object]) -> None:
        """写入官方 OpenAI Chat Completions reasoning 参数。"""
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        elif not self.thinking:
            params["reasoning_effort"] = "none"


def _warn_chat_builtin_tools(tools: tuple[ToolDefinition, ...]) -> None:
    """提示 Chat Completions 不支持 Responses 内建工具。"""
    for tool in tools:
        if tool.builtin is None:
            continue
        _LOGGER.warning(
            "OpenAI Chat Completions does not support builtin tool %r "
            "with type=%r; builtin tools are not available",
            tool.name,
            tool.builtin.get("type"),
        )
