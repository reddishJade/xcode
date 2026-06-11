"""OpenAI Chat Completions 兼容 provider 基类。

DeepSeek、ChatGLM、MiMo 和 OpenAIChat 共享的客户端创建、
流式调用和指标记录模式提取到此类中。子类只需实现 _stream_sync
和 provider 专有的参数构建逻辑。
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition

from .codec import normalize_cross_provider_messages, to_chat_messages, to_chat_tool
from .metrics import ProviderMetricsMixin
from .runtime import ProviderRuntime
from .stream_codec import chat_stream_to_events


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
        transport: str = "openai_chat",
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI as _OpenAIClient

            client = _OpenAIClient(api_key=api_key, base_url=base_url)
        self.client = client
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.runtime = runtime or ProviderRuntime()
        self.transport = transport
        self._current_options: StreamOptions | None = None
        self._ensure_metrics()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        self._current_options = options
        messages = self._normalize_messages(messages)
        for event in self._stream_sync(messages, tuple(tools), **kwargs):
            yield event

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **kwargs: Any,
    ) -> Iterator[ProviderEvent]:
        self._warn_builtin_tools(tools)
        clean = self._clean_reasoning_content(messages)
        api_messages = to_chat_messages(clean)
        params = self._build_chat_params(api_messages, tools, **kwargs)
        yield from self._call_chat_api(params, len(api_messages))

    def _warn_builtin_tools(self, tools: tuple[ToolDefinition, ...]) -> None:
        """子类可覆写以对不支持的 builtin 工具发出警告。"""

    def _build_chat_params(
        self,
        api_messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构建 Chat Completions 请求参数。

        子类可覆写以添加 provider 专有字段（如 response_format、tool_stream）。
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "tools": [to_chat_tool(t.name, t.description, t.parameters) for t in tools],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self._build_thinking_params(params)
        return params

    def _call_chat_api(
        self,
        params: dict[str, Any],
        message_count: int,
    ) -> Iterator[ProviderEvent]:
        """调用 openai.OpenAI.chat.completions.create 并通过 _intercept_usage 流式返回事件。"""
        opts = getattr(self, "_current_options", None)

        # 请求级覆盖
        if opts:
            if opts.api_key:
                params["api_key"] = opts.api_key
            extra_headers: dict[str, str] = {}
            if opts.session_id:
                extra_headers["x-session-id"] = opts.session_id
            if opts.headers:
                extra_headers.update(opts.headers)
            if extra_headers:
                params["extra_headers"] = extra_headers

        stream = self.runtime.run(lambda: self.client.chat.completions.create(**params))
        return chat_stream_to_events(self._intercept_stream(stream, message_count))

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

        opts = getattr(self, "_current_options", None)
        if opts and opts.thinking_budgets and effective:
            level_name = getattr(opts, "thinking_level", None)
            if level_name and level_name != "off":
                budget = getattr(opts.thinking_budgets, level_name, None)
                if budget and budget > 0:
                    extra.setdefault("thinking", {})["budget_tokens"] = budget

        if extra:
            existing = params.get("extra_body", {})
            existing.update(extra)
            params["extra_body"] = existing
        if effective and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """跨 provider 消息归一化。

        当消息来自不同 provider 时（如 DeepSeek → MiMo），
        将 provider 专有字段（如 reasoning_content）转为通用文本格式。
        子类可重写以添加额外转换。
        """
        return normalize_cross_provider_messages(messages, self.transport)

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
