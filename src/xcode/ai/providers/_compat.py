"""OpenAI Chat Completions 兼容 provider 基类。

所有使用 OpenAI Chat Completions API 的 provider（OpenAI、DeepSeek、
ChatGLM、MiMo）共享此基类。子类只需覆写 _stream_sync 以定制 API 调用。
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any

from xcode.ai.cache import extract_cache_usage
from xcode.ai.events import ProviderEvent
from xcode.ai.types import ProviderConfig, StreamOptions, ThinkingBudgets, ToolDefinition

from ._codec import normalize_cross_provider_messages, to_chat_messages, to_chat_tools
from ._runtime import ProviderRuntime
from ._stream import chat_stream_to_events


def _lookup_thinking_budget(budgets: ThinkingBudgets, level_name: str) -> int | None:
    if level_name == "minimal":
        return budgets.minimal
    if level_name == "low":
        return budgets.low
    if level_name == "medium":
        return budgets.medium
    if level_name == "high":
        return budgets.high
    if level_name == "xhigh":
        return budgets.xhigh
    return None


class OpenAICompatProvider:
    """OpenAI Chat Completions 兼容基类。

    子类必须设置:
    - self.transport（在 super().__init__ 之前或通过 transport 参数）
    - _stream_sync(messages, tools, ...) 实现

    子类可覆写:
    - _build_thinking_params() 构建 provider 专有 thinking 参数
    - _clean_reasoning_content() 控制 reasoning_content 保留策略
    - _record_usage() 记录 provider 专有指标
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        transport: str = "openai_chat",
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI as _OpenAIClient

            client = _OpenAIClient(
                api_key=config.api_key, base_url=config.base_url
            )
        self.client = client
        self.config = config
        self.runtime = ProviderRuntime()
        self.transport = transport
        self._current_options: StreamOptions | None = None
        self._metrics: dict[str, object] = self._init_metrics()

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def thinking(self) -> bool:
        return self.config.thinking

    @property
    def reasoning_effort(self) -> str | None:
        return self.config.reasoning_effort

    @property
    def metrics(self) -> dict[str, object]:
        return self._metrics

    def _init_metrics(self) -> dict[str, object]:
        return {
            "transport": self.transport,
            "sent_messages": 0,
            "cached_tokens": 0,
            "cache_hit_rate": 0.0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
        }

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        self._current_options = options
        messages = self._normalize_messages(messages)
        for event in self._stream_sync(messages, tuple(tools)):
            yield event

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> Iterator[ProviderEvent]:
        self._warn_builtin_tools(tools)
        clean = self._clean_reasoning_content(messages)
        api_messages = to_chat_messages(clean)
        params = self._build_chat_params(api_messages, tools)
        yield from self._call_chat_api(params, len(api_messages))

    def _warn_builtin_tools(self, tools: tuple[ToolDefinition, ...]) -> None:
        """子类可覆写以对不支持的 builtin 工具发出警告。"""

    def _build_chat_params(
        self,
        api_messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> dict[str, Any]:
        """构建 Chat Completions 请求参数。子类可覆写添加专有字段。"""
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "tools": to_chat_tools(tools),
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
        """调用 OpenAI chat.completions.create 并流式返回事件。"""
        opts = self._current_options

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

        stream = self.runtime.run(
            lambda: self.client.chat.completions.create(**params)
        )
        intercepted = self._intercept_usage(stream, message_count)
        return chat_stream_to_events(intercepted)

    # ── metrics 拦截 ──

    def _intercept_usage(
        self,
        chunks: Iterator[Any],
        message_count: int,
    ) -> Iterator[Any]:
        """拦截流式响应，在遇到 usage 时记录指标。"""
        for chunk in chunks:
            usage = getattr(chunk, "usage", None)
            if usage:
                self._record_usage(chunk, message_count)
            yield chunk

    def _record_usage(self, response: Any, sent_messages: int) -> None:
        """记录 usage 指标。子类可覆写以添加 provider 专有字段。"""
        self._metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            cache_usage = extract_cache_usage(response)
            self._metrics["cached_tokens"] = cache_usage.hit_tokens
            self._metrics["cache_hit_rate"] = cache_usage.hit_rate
            self._metrics["prompt_cache_hit_tokens"] = cache_usage.hit_tokens
            self._metrics["prompt_cache_miss_tokens"] = cache_usage.miss_tokens

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self._metrics["reasoning_tokens"] = reasoning or 0

    # ── thinking 参数构建 ──

    def _build_thinking_extra_body(
        self, thinking_override: bool | None = None
    ) -> dict[str, Any]:
        """构建 provider-agnostic thinking extra_body。"""
        effective = (
            self.config.thinking
            if thinking_override is None
            else thinking_override
        )
        if effective:
            return {"thinking": {"type": "enabled"}}
        return {"thinking": {"type": "disabled"}}

    def _build_thinking_params(
        self,
        params: dict[str, Any],
        thinking_override: bool | None = None,
    ) -> None:
        """将 thinking 配置写入 params（extra_body 和 reasoning_effort）。"""
        effective = (
            self.config.thinking
            if thinking_override is None
            else thinking_override
        )
        extra = self._build_thinking_extra_body(thinking_override)

        opts = self._current_options
        if opts and opts.thinking_budgets and effective:
            level_name = opts.thinking_level
            if level_name and level_name != "off":
                budget = _lookup_thinking_budget(
                    opts.thinking_budgets, level_name
                )
                if budget and budget > 0:
                    extra.setdefault("thinking", {})["budget_tokens"] = budget

        if extra:
            existing = params.get("extra_body", {})
            existing.update(extra)
            params["extra_body"] = existing
        if effective and self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort

    # ── 消息归一化 ──

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """跨 provider 消息归一化。子类可重写以添加额外转换。"""
        return normalize_cross_provider_messages(messages, self.transport)

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """从消息历史中清理 reasoning_content。

        子类可重写以实现不同清理策略。
        默认行为：当 thinking 关闭时删除所有 reasoning_content。
        """
        if not messages or self.config.thinking:
            return messages
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            msg.pop("reasoning_content", None)
        return cleaned
