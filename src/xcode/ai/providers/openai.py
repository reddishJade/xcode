from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from ...agent.types import ToolDefinition
from ...harness.agent_runtime.events import ProviderEvent

from .codec import (
    chat_stream_to_events,
    responses_stream_to_events,
    to_chat_messages,
    to_chat_tool,
    to_responses_input,
    to_responses_tool,
)
from .runtime import ProviderRuntime


class OpenAIChatProvider:
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
        prompt_cache_key: str | None = None,
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: openai. Install project requirements first."
                ) from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.runtime = runtime or ProviderRuntime()
        self.prompt_cache_key = prompt_cache_key
        self.transport = "openai_chat"
        self.metrics: dict[str, object] = {
            "transport": self.transport,
            "previous_response_id": None,
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
        # 使用 to_chat_messages 转换为 Chat Completions 格式
        chat_messages = to_chat_messages(messages)
        kwargs: dict[str, object] = {
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
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        def intercept_usage(chunks):
            for chunk in chunks:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._record_usage(chunk, len(chat_messages))
                yield chunk

        stream = self.runtime.run(lambda: self.client.chat.completions.create(**kwargs))
        self._ensure_metrics()
        self.metrics["sent_messages"] = len(chat_messages)
        yield from chat_stream_to_events(intercept_usage(stream))

    def _record_usage(self, response, sent_messages: int) -> None:
        self._ensure_metrics()
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            # cached_tokens
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            self.metrics["cached_tokens"] = cached or 0

            # reasoning_tokens
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
            self.metrics["reasoning_tokens"] = reasoning or 0

    def _ensure_metrics(self) -> None:
        if not hasattr(self, "metrics"):
            self.metrics = {
                "transport": getattr(self, "transport", "openai_chat"),
                "previous_response_id": None,
                "sent_messages": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            }


class OpenAIResponsesProvider:
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
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: openai. Install project requirements first."
                ) from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.runtime = runtime or ProviderRuntime()
        self.prompt_cache_key = prompt_cache_key
        self.transport = "openai_responses"
        self.previous_response_id: str | None = None
        self._last_sent_message_index: int = 0
        self._pending_sent_message_index: int = 0
        self.metrics: dict[str, object] = {
            "transport": self.transport,
            "previous_response_id": None,
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
        kwargs = self._responses_kwargs(messages, tools, stream=True)
        stream = self.runtime.run(lambda: self.client.responses.create(**kwargs))
        self.metrics["sent_messages"] = len(kwargs["input"])

        def intercept_events(events: Iterable[object]) -> Iterator[object]:
            """拦截流以记录 metadata，同时保持 streaming 不缓存。"""
            for raw_event in events:
                if str(getattr(raw_event, "type", "")).endswith(".completed"):
                    response = getattr(raw_event, "response", None)
                    if response is not None:
                        response_id = getattr(response, "id", None)
                        if response_id:
                            self.previous_response_id = str(response_id)
                            self.metrics["previous_response_id"] = self.previous_response_id
                            self._last_sent_message_index = self._pending_sent_message_index
                        self._record_usage(response, len(kwargs["input"]))
                yield raw_event

        yield from responses_stream_to_events(intercept_events(stream))

    def _responses_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        stream: bool,
    ) -> dict:
        # 按 raw message index 切片（不是 converted item index），
        # 这样第二轮工具结果只发 function_call_output，不会重复发 assistant 的 function_call。
        if self.previous_response_id is None:
            raw_input_messages = messages
        else:
            raw_input_messages = messages[self._last_sent_message_index:]

        converted = to_responses_input(raw_input_messages)

        # stateful 模式下过滤掉模型自己产生的 assistant items。
        # previous_response_id 已经承接上一轮 response，
        # 下一轮只需要追加函数结果或新的用户输入。
        if self.previous_response_id is not None:
            converted = [
                item for item in converted
                if item.get("type") == "function_call_output"
                or item.get("role") in {"user", "system", "developer"}
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
                )
                for t in tools
            ],
            "stream": stream,
        }

        # reasoning 配置（Responses API 使用 reasoning 参数）
        # 只在显式配置时传 reasoning 参数
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        elif not self.thinking:
            kwargs["reasoning"] = {"effort": "none"}

        if self.previous_response_id is not None:
            kwargs["previous_response_id"] = self.previous_response_id
        if self.prompt_cache_key:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
        return kwargs

    def _record_usage(self, response, sent_messages: int) -> None:
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if not usage:
            return

        # Responses API 使用 input_tokens_details / output_tokens_details
        input_details = getattr(usage, "input_tokens_details", None)
        cached = getattr(input_details, "cached_tokens", 0) if input_details else 0
        self.metrics["cached_tokens"] = cached or 0

        output_details = getattr(usage, "output_tokens_details", None)
        reasoning = getattr(output_details, "reasoning_tokens", 0) if output_details else 0
        self.metrics["reasoning_tokens"] = reasoning or 0
