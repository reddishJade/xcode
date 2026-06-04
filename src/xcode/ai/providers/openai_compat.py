from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from xcode.ai.events import ProviderEvent
from xcode.ai.types import ToolDefinition

from .metrics import ProviderMetricsMixin
from .runtime import ProviderRuntime
from .stream_codec import chat_stream_to_events

"""OpenAI Chat Completions 兼容 provider 基类。

DeepSeek、ChatGLM、MiMo 和 OpenAIChat 共享的客户端创建、
流式调用和指标记录模式提取到此类中。子类只需实现 _stream_sync
和 provider 专有的参数构建逻辑。
"""


class OpenAICompatProvider(ProviderMetricsMixin):
    """OpenAI Chat Completions 兼容基类。

    子类必须设置:
    - self.transport（在 super().__init__ 之前或通过 transport 参数）
    - _stream_sync(messages, tools, ...) 实现

    子类可覆写:
    - _build_chat_kwargs() 构建 provider 专有参数
    - _record_usage() 记录 provider 专有指标
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        thinking: bool = True,
        reasoning_effort: str | None = None,
        runtime: ProviderRuntime | None = None,
        client: Any | None = None,
        transport: str = "openai_chat",
        import_error_msg: str = "Missing dependency: openai.",
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(import_error_msg) from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.runtime = runtime or ProviderRuntime()
        self.transport = transport
        self._ensure_metrics()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        for event in self._stream_sync(messages, tuple(tools), **kwargs):
            yield event

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **kwargs: Any,
    ) -> Iterator[ProviderEvent]:
        raise NotImplementedError

    def _call_chat_api(
        self,
        params: dict[str, Any],
        message_count: int,
    ) -> Iterator[ProviderEvent]:
        """调用 chat.completions.create 并通过 _intercept_usage 流式返回事件。"""
        create = cast(Any, self.client.chat.completions.create)
        stream = self.runtime.run(lambda: create(**params))
        self._ensure_metrics()
        self.metrics["sent_messages"] = message_count
        return chat_stream_to_events(self._intercept_usage(stream, message_count))