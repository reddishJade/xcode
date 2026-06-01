from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Iterable
from typing import Any

from ...agent.types import ToolDefinition
from ...harness.agent_runtime.events import ProviderEvent

from .codec import chat_stream_to_events, to_chat_messages, to_chat_tool
from .runtime import ProviderRuntime

"""Xiaomi MiMo provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。

API 文档：https://platform.xiaomimimo.com/
支持模型：mimo-v2.5-pro、mimo-v2.5、mimo-v2-flash 等。
"""

# MiMo API 地址
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


class MiMoProvider:
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
        self.metrics: dict[str, object] = {
            "transport": self.transport,
            "sent_messages": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

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
            "tools": [
                to_chat_tool(t.name, t.description, t.schema) for t in tools
            ],
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

        def intercept_usage(chunks: Iterable[Any]) -> Iterator[Any]:
            for chunk in chunks:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._record_usage(chunk, len(openai_messages))
                yield chunk

        stream = self.runtime.run(
            lambda: self.client.chat.completions.create(**kwargs)
        )
        self.metrics["sent_messages"] = len(openai_messages)
        yield from chat_stream_to_events(intercept_usage(stream))

    def _record_usage(self, response, sent_messages: int) -> None:
        """记录 usage 指标，包含缓存 Token 和 reasoning_tokens 统计。"""
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            # 缓存 Token
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            self.metrics["cached_tokens"] = cached or 0

            # reasoning_tokens
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
            self.metrics["reasoning_tokens"] = reasoning or 0
