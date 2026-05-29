from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from ...agent.types import ToolDefinition
from ...harness.agent_runtime.events import ProviderEvent

from .codec import chat_stream_to_events, to_chat_tool, to_openai_messages
from .runtime import ProviderRuntime

"""Xiaomi MiMo provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。"""


class MiMoProvider:
    """Xiaomi MiMo Chat API 适配。

    和 DeepSeekProvider 基本一致，使用 OpenAI 兼容接口，额外处理 reasoning_content 字段。
    支持 mimo-v2.5-pro（复杂推理）和 mimo-v2.5（多模态）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        thinking: bool = True,
        reasoning_effort: str | None = None,
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
        self.reasoning_effort = reasoning_effort
        self.runtime = runtime or ProviderRuntime()
        self.transport = "chat_completions"
        self.metrics: dict[str, object] = {
            "transport": self.transport,
            "sent_messages": 0,
            "cached_tokens": 0,
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
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "tools": [
                to_chat_tool(
                    t.name,
                    t.description,
                    t.schema,
                )
                for t in tools
            ],
            "stream": True,
        }
        extra_body = {}
        if not self.thinking:
            extra_body["thinking"] = {"type": "disabled"}
        elif self.reasoning_effort:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.thinking and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        stream = self.runtime.run(lambda: self.client.chat.completions.create(**kwargs))  # pyright: ignore
        yield from chat_stream_to_events(stream)
