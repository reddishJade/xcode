from __future__ import annotations

import copy
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
    - _build_thinking_extra_body() 构建 provider 专有 thinking 参数
    - _clean_reasoning_content() 控制 reasoning_content 保留策略
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

    # --- Unified thinking abstractions ---

    def _build_thinking_extra_body(
        self, thinking_override: bool | None = None
    ) -> dict[str, Any]:
        """构建 provider-agnostic thinking extra_body。

        返回形如 {"thinking": {"type": "enabled"|"disabled"}} 的 dict。
        子类可重写以添加 provider 专有字段。
        """
        effective = self.thinking if thinking_override is None else thinking_override
        if effective:
            return {"thinking": {"type": "enabled"}}
        return {"thinking": {"type": "disabled"}}

    def _build_thinking_params(
        self,
        params: dict[str, Any],
        thinking_override: bool | None = None,
    ) -> None:
        """将 thinking 配置写入 params（extra_body 和 reasoning_effort）。

        子类应在其 _stream_sync 中调用此方法。
        """
        effective = self.thinking if thinking_override is None else thinking_override
        extra = self._build_thinking_extra_body(thinking_override)
        if extra:
            existing = params.get("extra_body", {})
            existing.update(extra)
            params["extra_body"] = existing
        if effective and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """从消息历史中清理 reasoning_content。

        子类可重写以实现不同清理策略（如 DeepSeek 需要在 tool loop 中保留）。
        默认行为：当 thinking 关闭时删除所有 reasoning_content。
        """
        if not messages or self.thinking:
            return messages
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            msg.pop("reasoning_content", None)
        return cleaned
