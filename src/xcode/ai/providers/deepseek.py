"""DeepSeek provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from xcode.ai.types import ProviderConfig, ToolDefinition

from ._codec import to_chat_messages, to_chat_tools
from ._compat import OpenAICompatProvider

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek Chat API 适配。"""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            config,
            transport="deepseek_chat",
            client=client,
        )
        self._metrics["prompt_cache_hit_tokens"] = 0
        self._metrics["prompt_cache_miss_tokens"] = 0

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> Iterator[Any]:
        cleaned_messages = self._clean_reasoning_content(messages)
        api_messages = to_chat_messages(cleaned_messages)

        strict_tools = self.config.extra.get("strict_tools", False)
        effective_format = self.config.response_format
        if effective_format and effective_format.get("type") == "json_object":
            api_messages = self._ensure_json_word(api_messages)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "tools": to_chat_tools(tools, strict=strict_tools),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if effective_format:
            params["response_format"] = effective_format

        self._build_thinking_params(params)

        yield from self._call_chat_api(params, len(api_messages))

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """清理 reasoning_content 以符合 DeepSeek API 要求。"""
        if not messages:
            return messages

        cleaned = copy.deepcopy(messages)
        in_tool_loop = cleaned[-1].get("role") == "tool"

        if not in_tool_loop:
            for msg in cleaned:
                msg.pop("reasoning_content", None)
        else:
            last_user_idx = -1
            for i in range(len(cleaned) - 1, -1, -1):
                if cleaned[i].get("role") == "user":
                    last_user_idx = i
                    break
            for i in range(last_user_idx):
                if cleaned[i].get("role") == "assistant":
                    cleaned[i].pop("reasoning_content", None)

        return cleaned

    def _ensure_json_word(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """确保消息中包含 'json' 关键字（DeepSeek API 约束）。"""
        has_json_word = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and "json" in content.lower():
                has_json_word = True
                break
            elif isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and "json" in part.get("text", "").lower()
                    ):
                        has_json_word = True
                        break
        if not has_json_word:
            messages = copy.deepcopy(messages)
            appended = False
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content")
                    if isinstance(content, str):
                        msg["content"] = (
                            content + "\nNote: Output must be in JSON format."
                        )
                        appended = True
                        break
            if not appended and messages:
                first_msg = messages[0]
                content = first_msg.get("content")
                if isinstance(content, str):
                    first_msg["content"] = (
                        content + "\nNote: Output must be in JSON format."
                    )
                elif isinstance(content, list):
                    content.append(
                        {
                            "type": "text",
                            "text": "Note: Output must be in JSON format.",
                        }
                    )
        return messages
