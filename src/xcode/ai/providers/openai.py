from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from xcode.ai.cache import tool_catalog_fingerprint
from xcode.ai.events import ProviderEvent
from xcode.ai.types import (
    CacheRetention,
    PromptCacheRetention,
    StreamOptions,
    ToolDefinition,
)

from .codec import to_chat_messages, to_chat_tool, to_responses_input, to_responses_tool
from .openai_compat import OpenAICompatProvider
from .stream_codec import responses_response_to_events, responses_stream_to_events
from .runtime import ProviderRuntime

_RESPONSES_OPTION_FIELDS = (
    "background",
    "context_management",
    "conversation",
    "include",
    "instructions",
    "max_tool_calls",
    "moderation",
    "parallel_tool_calls",
    "prompt",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "store",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
)

_RESPONSES_INPUT_TOKEN_COUNT_FIELDS = frozenset(
    {
        "conversation",
        "input",
        "instructions",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "reasoning",
        "text",
        "tool_choice",
        "tools",
        "truncation",
    }
)

_LOGGER = logging.getLogger(__name__)

_CACHE_RETENTION_TO_PROMPT_CACHE_RETENTION: dict[
    CacheRetention,
    PromptCacheRetention,
] = {
    "short": "in_memory",
    "long": "24h",
}

_SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "<system-prompt-dynamic-boundary />"


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
        client=None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            runtime=runtime,
            client=client,
            transport="openai_chat",
            import_error_msg="Missing dependency: openai. Install project requirements first.",
        )
        self.response_format = response_format

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[ProviderEvent]:
        chat_messages = to_chat_messages(messages)
        _warn_chat_builtin_tools(tools)
        params: dict[str, object] = {
            "model": self.model,
            "messages": chat_messages,
            "tools": [
                to_chat_tool(
                    t.name,
                    t.description,
                    t.schema,
                )
                for t in tools
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        effective_format = response_format or self.response_format
        if effective_format:
            params["response_format"] = effective_format
        self._build_openai_reasoning_params(params)

        yield from self._call_chat_api(params, len(chat_messages))

    def _build_openai_reasoning_params(self, params: dict[str, object]) -> None:
        """写入官方 OpenAI Chat Completions reasoning 参数。"""
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        elif not self.thinking:
            params["reasoning_effort"] = "none"


class OpenAIResponsesProvider(OpenAICompatProvider):
    """OpenAI Responses API provider（stateful 模式）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        thinking: bool = True,
        reasoning_effort: str | None = None,
        runtime: ProviderRuntime | None = None,
        prompt_cache_key: str | None = None,
        response_format: dict[str, Any] | None = None,
        client=None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            runtime=runtime,
            client=client,
            transport="openai_responses",
            import_error_msg="Missing dependency: openai. Install project requirements first.",
        )
        self.prompt_cache_key = prompt_cache_key
        self.previous_response_id: str | None = None
        self._last_sent_message_index: int = 0
        self._pending_sent_message_index: int = 0
        self._pending_store_response: bool = True
        self._stateless_persisted_items: list[dict[str, Any]] = []
        self.response_format = response_format
        self.metrics["previous_response_id"] = None

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

    async def retrieve_response(
        self,
        response_id: str,
        *,
        stream: bool = False,
        include: list[str] | None = None,
        starting_after: int | None = None,
        include_obfuscation: bool | None = None,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """获取后台 Responses 响应，支持轮询或从指定序号恢复流。"""
        self._current_options = options
        for event in self._retrieve_response_sync(
            response_id,
            stream=stream,
            include=include,
            starting_after=starting_after,
            include_obfuscation=include_obfuscation,
        ):
            yield event

    def _retrieve_response_sync(
        self,
        response_id: str,
        *,
        stream: bool,
        include: list[str] | None,
        starting_after: int | None,
        include_obfuscation: bool | None,
    ) -> Iterator[ProviderEvent]:
        """调用 Responses retrieve 并转换为统一事件。"""
        params = _responses_retrieve_kwargs(
            getattr(self, "_current_options", None),
            stream=stream,
            include=include,
            starting_after=starting_after,
            include_obfuscation=include_obfuscation,
        )
        client = self._responses_client()
        retrieve = cast(Any, client.responses.retrieve)
        response_or_stream = self.runtime.run(lambda: retrieve(response_id, **params))

        if stream:
            yield from responses_stream_to_events(
                cast(
                    Any,
                    self._intercept_responses_stream(
                        cast(Any, response_or_stream),
                        0,
                        on_response_completed=self._record_completed_response,
                    ),
                )
            )
            return

        self._record_retrieved_response(response_or_stream)
        yield from responses_response_to_events(cast(Any, response_or_stream))

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[ProviderEvent]:
        params = self._responses_kwargs(messages, tools, stream=True)
        effective_format = response_format or self.response_format
        if effective_format:
            params["text"] = _responses_text_config(effective_format)
        self._apply_responses_options(params)
        client = self._responses_client()
        self._record_input_token_count(client, params)
        create = cast(Any, client.responses.create)
        stream = self.runtime.run(lambda: create(**params))
        message_count = len(params["input"])
        yield from responses_stream_to_events(
            cast(
                Any,
                self._intercept_responses_stream(
                    stream,
                    message_count,
                    on_response_completed=self._record_completed_response,
                ),
            )
        )

    def _record_completed_response(self, response: object) -> None:
        """记录 Responses API 的 stateful 游标。"""
        self._stateless_persisted_items = _persistable_output_items(response)
        if not self._pending_store_response:
            self.previous_response_id = None
            self.metrics["previous_response_id"] = None
            self._last_sent_message_index = self._pending_sent_message_index
            return

        response_id = getattr(response, "id", None)
        if not response_id:
            return
        self.previous_response_id = str(response_id)
        self.metrics["previous_response_id"] = self.previous_response_id
        self._last_sent_message_index = self._pending_sent_message_index

    def _record_retrieved_response(self, response: object) -> None:
        """记录后台 retrieve 返回的状态和可续接响应。"""
        self._ensure_metrics()
        self.metrics["sent_messages"] = 0
        response_id = getattr(response, "id", None)
        if response_id:
            self.metrics["background_response_id"] = str(response_id)

        raw_status = getattr(response, "status", None)
        status = str(raw_status) if raw_status else None
        if status:
            self.metrics["background_response_status"] = status
        if status in (None, "completed"):
            self._record_completed_response(response)
        self._record_usage(response, 0)

    def _responses_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        stream: bool,
    ) -> dict:
        if self.previous_response_id is None:
            raw_input_messages = messages
        else:
            raw_input_messages = messages[self._last_sent_message_index :]

        instructions, input_messages = _extract_responses_instructions(
            raw_input_messages
        )
        converted = to_responses_input(input_messages)
        if self.previous_response_id is None and self._stateless_persisted_items:
            converted = [*self._stateless_persisted_items, *converted]

        if self.previous_response_id is not None:
            instructions = None
            converted = [
                item for item in converted if _is_incremental_responses_input(item)
            ]

        self._pending_sent_message_index = len(messages)

        kwargs: dict[str, object] = {
            "model": self.model,
            "input": converted,
            "tools": [
                to_responses_tool(
                    t.name,
                    t.description,
                    t.schema,
                    t.builtin,
                    strict=True,
                )
                for t in tools
            ],
            "stream": stream,
        }

        reasoning: dict[str, object] = {}
        if self.reasoning_effort:
            reasoning["effort"] = self.reasoning_effort
        elif not self.thinking:
            reasoning["effort"] = "none"
        opts = getattr(self, "_current_options", None)
        if opts and opts.reasoning_summary:
            reasoning["summary"] = opts.reasoning_summary
        if reasoning:
            kwargs["reasoning"] = reasoning

        if self.previous_response_id is not None:
            kwargs["previous_response_id"] = self.previous_response_id
        if instructions:
            kwargs["instructions"] = instructions
        prompt_cache_key = _responses_prompt_cache_key(self.prompt_cache_key, tools)
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key
        return kwargs

    def _apply_responses_options(self, params: dict[str, object]) -> None:
        """将通用 StreamOptions 映射到 Responses API 请求参数。"""
        opts = getattr(self, "_current_options", None)
        if opts is None:
            return

        extra_headers: dict[str, str] = {}
        if opts.session_id:
            extra_headers["x-session-id"] = opts.session_id
        if opts.headers:
            extra_headers.update(opts.headers)
        if extra_headers:
            params["extra_headers"] = extra_headers
        if opts.metadata and "metadata" not in params:
            params["metadata"] = opts.metadata
        if opts.temperature is not None:
            params["temperature"] = opts.temperature
        if opts.max_tokens is not None:
            params["max_output_tokens"] = opts.max_tokens
        if opts.timeout_ms is not None:
            params["timeout"] = opts.timeout_ms / 1000
        for field_name in _RESPONSES_OPTION_FIELDS:
            value = getattr(opts, field_name)
            if value is not None:
                if field_name == "instructions":
                    params[field_name] = _merge_responses_instructions(
                        params.get("instructions"), value
                    )
                else:
                    params[field_name] = value
        if (
            opts.server_compact_threshold is not None
            and "context_management" not in params
        ):
            context_management = _responses_compaction_context_management(
                opts.server_compact_threshold
            )
            if context_management:
                params["context_management"] = context_management
        # cache_retention 映射到 prompt_cache_retention，显式设置优先
        if "prompt_cache_retention" not in params:
            retention = _map_cache_retention(opts.cache_retention)
            if retention is not None:
                params["prompt_cache_retention"] = retention
        if opts.verbosity is not None:
            existing_text = params.get("text")
            text_config = (
                dict(cast(dict[str, object], existing_text))
                if isinstance(existing_text, dict)
                else {}
            )
            text_config["verbosity"] = opts.verbosity
            params["text"] = text_config
        if opts.response_extra_params:
            for key, value in opts.response_extra_params.items():
                if value is not None and key not in params:
                    params[key] = value
        self._ensure_stateless_reasoning_options(params)
        self._pending_store_response = params.get("store") is not False

    def _responses_client(self) -> Any:
        """按请求级 API key 返回 Responses 客户端。"""
        opts = getattr(self, "_current_options", None)
        if opts is None or not opts.api_key:
            return self.client
        return self.client.with_options(api_key=opts.api_key)

    def _record_input_token_count(
        self,
        client: Any,
        params: dict[str, object],
    ) -> None:
        """调用 Responses input_tokens 接口记录官方输入 token 数。"""
        try:
            count = cast(Any, client.responses.input_tokens.count)
        except AttributeError as error:
            self.metrics["input_token_count_error"] = str(error)
            _LOGGER.warning("OpenAI Responses input token count API is unavailable")
            return

        count_kwargs = _responses_input_token_count_kwargs(params)
        try:
            response = self.runtime.run(lambda: count(**count_kwargs))
        except RuntimeError as error:
            self.metrics["input_token_count_error"] = str(error)
            _LOGGER.warning(
                "OpenAI Responses input token count failed: %s",
                error,
            )
            return

        input_tokens = getattr(response, "input_tokens", None)
        if input_tokens is None:
            self.metrics["input_token_count_error"] = "missing input_tokens"
            _LOGGER.warning("OpenAI Responses input token count response is missing")
            return
        self.metrics["input_tokens"] = int(input_tokens)

    def _ensure_stateless_reasoning_options(self, params: dict[str, object]) -> None:
        """store=false 时请求 encrypted reasoning 以便后续回灌。"""
        if params.get("store") is not False or not self.thinking:
            return
        raw_include = params.get("include")
        include = (
            list(cast(list[str], raw_include)) if isinstance(raw_include, list) else []
        )
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        params["include"] = include

    def _record_usage(self, response, sent_messages: int) -> None:
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if not usage:
            return

        input_details = getattr(usage, "input_tokens_details", None)
        cached = getattr(input_details, "cached_tokens", 0) if input_details else 0
        self.metrics["cached_tokens"] = cached or 0

        output_details = getattr(usage, "output_tokens_details", None)
        reasoning = (
            getattr(output_details, "reasoning_tokens", 0) if output_details else 0
        )
        self.metrics["reasoning_tokens"] = reasoning or 0


def _map_cache_retention(
    cache_retention: CacheRetention,
) -> PromptCacheRetention | None:
    """将高层 CacheRetention 语义映射到 OpenAI prompt_cache_retention 参数值。

    映射规则：
    - "none" → None（不发送 prompt_cache_retention，使用 API 默认行为）
    - "short" → "in_memory"
    - "long" → "24h"
    """
    return _CACHE_RETENTION_TO_PROMPT_CACHE_RETENTION.get(cache_retention)


def _responses_text_config(response_format: dict[str, Any]) -> dict[str, object]:
    """将 Chat 风格 response_format 转为 Responses text 配置。"""
    if response_format.get("type") != "json_schema":
        return {"format": response_format}

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return {"format": response_format}

    flattened = dict(json_schema)
    flattened["type"] = "json_schema"
    return {"format": flattened}


def _responses_input_token_count_kwargs(
    params: dict[str, object],
) -> dict[str, object]:
    """从 Responses create 参数提取 input_tokens.count 支持的字段。"""
    result = {
        field_name: params[field_name]
        for field_name in _RESPONSES_INPUT_TOKEN_COUNT_FIELDS
        if field_name in params
    }
    for field_name in ("extra_headers", "timeout"):
        if field_name in params:
            result[field_name] = params[field_name]
    return result


def _responses_compaction_context_management(
    compact_threshold: int,
) -> list[dict[str, object]]:
    """构造 Responses 服务端压缩配置。"""
    if compact_threshold <= 0:
        return []
    return [
        {
            "type": "compaction",
            "compact_threshold": compact_threshold,
        }
    ]


def _responses_retrieve_kwargs(
    opts: StreamOptions | None,
    *,
    stream: bool,
    include: list[str] | None,
    starting_after: int | None,
    include_obfuscation: bool | None,
) -> dict[str, object]:
    """构造 Responses retrieve 请求参数。"""
    result: dict[str, object] = {"stream": stream}
    effective_include = include
    if effective_include is None and opts is not None:
        effective_include = opts.include
    if effective_include is not None:
        result["include"] = effective_include
    if starting_after is not None:
        result["starting_after"] = starting_after
    if include_obfuscation is not None:
        result["include_obfuscation"] = include_obfuscation
    if opts is None:
        return result

    extra_headers: dict[str, str] = {}
    if opts.session_id:
        extra_headers["x-session-id"] = opts.session_id
    if opts.headers:
        extra_headers.update(opts.headers)
    if extra_headers:
        result["extra_headers"] = extra_headers
    if opts.timeout_ms is not None:
        result["timeout"] = opts.timeout_ms / 1000
    return result


def _responses_prompt_cache_key(
    base_key: str | None,
    tools: tuple[ToolDefinition, ...],
) -> str | None:
    """将工具目录指纹接入 Responses prompt_cache_key。"""
    if not base_key and not tools:
        return None

    fingerprint = tool_catalog_fingerprint(list(tools))
    tool_key = f"tools:{fingerprint}"
    if not base_key:
        return tool_key
    return f"{base_key}:{tool_key}"


def _extract_responses_instructions(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """将稳定 system/developer 前缀移到 Responses instructions 参数。"""
    instruction_parts: list[str] = []
    input_messages: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if (
            role in {"system", "developer"}
            and isinstance(content, str)
            and _SYSTEM_PROMPT_DYNAMIC_BOUNDARY in content
        ):
            stable, dynamic = content.split(_SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1)
            stable = stable.strip()
            dynamic = dynamic.strip()
            if stable:
                instruction_parts.append(stable)
            if dynamic:
                dynamic_message = dict(message)
                dynamic_message["content"] = dynamic
                input_messages.append(dynamic_message)
            continue
        input_messages.append(message)

    instructions = "\n\n".join(instruction_parts)
    return instructions or None, input_messages


def _merge_responses_instructions(
    existing: object,
    explicit: object,
) -> str:
    """合并运行期稳定指令和显式请求指令。"""
    parts: list[str] = []
    for item in (existing, explicit):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _is_incremental_responses_input(item: dict[str, Any]) -> bool:
    """保留 previous_response_id 后仍需显式回传的输入项。"""
    if item.get("type") in {"function_call_output", "shell_call_output"}:
        return True
    return item.get("role") == "user"


def _warn_chat_builtin_tools(tools: tuple[ToolDefinition, ...]) -> None:
    """提示 Chat Completions 不支持 Responses 内建工具。"""
    for tool in tools:
        if tool.builtin is None:
            continue
        _LOGGER.warning(
            "OpenAI Chat Completions does not support builtin tool %r "
            "with type=%r; use the Responses API for builtin tools",
            tool.name,
            tool.builtin.get("type"),
        )


def _persistable_output_items(response: object) -> list[dict[str, Any]]:
    """从 Responses 输出中提取可回灌的 item（reasoning / function_call）。"""
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return []

    items: list[dict[str, Any]] = []
    for item in output:
        serialized = _serialize_response_output_item(item)
        if serialized.get("type") in ("reasoning", "function_call"):
            items.append(serialized)
    return items


def _serialize_response_output_item(item: object) -> dict[str, Any]:
    """将 SDK 输出对象转换为 input 可复用的 dict。"""
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped

    result: dict[str, Any] = {"type": str(getattr(item, "type", ""))}
    for field_name in (
        "id",
        "summary",
        "encrypted_content",
        "content",
        "call_id",
        "name",
        "arguments",
    ):
        value = getattr(item, field_name, None)
        if value is not None:
            result[field_name] = value
    return result
