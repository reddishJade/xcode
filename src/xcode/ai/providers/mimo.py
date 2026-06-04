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
        self, messages: list[dict[str, Any]], tools: tuple[ToolDefinition, ...], **kwargs: Any
    ) -> Iterator[Any]:
        openai_messages = to_chat_messages(messages)

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": openai_messages,
            "tools": [to_chat_tool(t.name, t.description, t.schema) for t in tools],
            "stream": True,
        }

        extra_body: dict[str, Any] = {}
        if not self.thinking:
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body

        yield from self._call_chat_api(kwargs, len(openai_messages))