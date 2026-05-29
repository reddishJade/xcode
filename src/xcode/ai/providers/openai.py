from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from ...agent.types import ToolDefinition
from ...harness.agent_runtime.events import ProviderEvent

from .codec import (
    chat_stream_to_events,
    responses_stream_to_events,
    to_chat_tool,
    to_openai_messages,
    to_responses_tool,
)
from .runtime import ProviderRuntime


class OpenAIChatProvider:
    """OpenAI Chat Completions provider (兼容所有 OpenAI API 兼容服务，含 DeepSeek)。"""

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
        self.transport = "chat_completions"
        self.metrics: dict[str, object] = {
            "transport": self.transport,
            "previous_response_id": None,
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
        prompt_cache_key = getattr(self, "prompt_cache_key", None)
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key
        extra_body = {}
        if not self.thinking:
            extra_body["thinking"] = {"type": "disabled"}
        elif self.reasoning_effort:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.thinking and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        stream = self.runtime.run(lambda: self.client.chat.completions.create(**kwargs))
        self._ensure_metrics()
        self.metrics["sent_messages"] = len(kwargs["messages"])  # type: ignore[arg-type]
        yield from chat_stream_to_events(stream)

    def _record_usage(self, response, sent_messages: int) -> None:
        self._ensure_metrics()
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = (
            getattr(input_details, "cached_tokens", 0) if input_details else 0
        )
        self.metrics["cached_tokens"] = cached_tokens or 0

    def _ensure_metrics(self) -> None:
        if not hasattr(self, "metrics"):
            self.metrics = {
                "transport": getattr(self, "transport", "chat_completions"),
                "previous_response_id": None,
                "sent_messages": 0,
                "cached_tokens": 0,
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
        self.transport = "responses_stateful"
        self.previous_response_id: str | None = None
        self.metrics: dict[str, object] = {
            "transport": self.transport,
            "previous_response_id": None,
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
        kwargs = self._responses_kwargs(messages, tools, stream=True)
        stream = self.runtime.run(lambda: self.client.responses.create(**kwargs))
        self.metrics["sent_messages"] = len(kwargs["input"])
        for raw_event in stream:
            if str(getattr(raw_event, "type", "")).endswith(".completed"):
                response = getattr(raw_event, "response", None)
                if response is not None:
                    response_id = getattr(response, "id", None)
                    if response_id:
                        self.previous_response_id = str(response_id)
                        self.metrics["previous_response_id"] = self.previous_response_id
                    self._record_usage(response, len(kwargs["input"]))
            yield from responses_stream_to_events((raw_event,))

    def _responses_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        stream: bool,
    ) -> dict:
        converted = to_openai_messages(messages)
        input_messages = (
            converted if self.previous_response_id is None else converted[-1:]
        )
        kwargs: dict[str, object] = {
            "model": self.model,
            "input": input_messages,
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
        if self.previous_response_id is not None:
            kwargs["previous_response_id"] = self.previous_response_id
        if self.prompt_cache_key:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
        return kwargs

    def _record_usage(self, response, sent_messages: int) -> None:
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = (
            getattr(input_details, "cached_tokens", 0) if input_details else 0
        )
        self.metrics["cached_tokens"] = cached_tokens or 0
