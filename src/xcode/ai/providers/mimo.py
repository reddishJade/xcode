from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from xcode.ai.events import ProviderEvent
from xcode.ai.types import ToolDefinition

from .codec import to_chat_messages, to_chat_tool
from .metrics import ProviderMetricsMixin
from .stream_codec import chat_stream_to_events
from .runtime import ProviderRuntime

"""Xiaomi MiMo provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。

API 文档：https://platform.xiaomimimo.com/
支持模型：mimo-v2.5-pro、mimo-v2.5、mimo-v2-flash 等。
"""

# MiMo API 地址
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


class MiMoProvider(ProviderMetricsMixin):
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
        runtime: ProviderRuntime | None = None,
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Missing dependency: openai.") from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.thinking = thinking
        self.runtime = runtime or ProviderRuntime()
        self.transport = "mimo_chat"
        self._ensure_metrics()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[ProviderEvent]:
        for event in self._stream_sync(messages, tuple(tools)):
            yield event

    def _stream_sync(
        self, messages: list[dict[str, Any]], tools: tuple[ToolDefinition, ...]
    ) -> Iterator[ProviderEvent]:
        # MiMo 建议保留所有历史 reasoning_content，不做清理
        openai_messages = to_chat_messages(messages)

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": openai_messages,
            "tools": [to_chat_tool(t.name, t.description, t.schema) for t in tools],
            "stream": True,
        }

        # thinking 配置
        extra_body: dict[str, Any] = {}
        if not self.thinking:
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body

        create = cast(Any, self.client.chat.completions.create)
        stream = self.runtime.run(lambda: create(**kwargs))
        self.metrics["sent_messages"] = len(openai_messages)
        yield from chat_stream_to_events(self._intercept_usage(stream, len(openai_messages)))
