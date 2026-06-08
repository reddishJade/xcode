from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from xcode.ai.events import ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition

from .codec import to_chat_messages, to_chat_tool, to_responses_input, to_responses_tool
from .openai_compat import OpenAICompatProvider
from .stream_codec import responses_stream_to_events
from .runtime import ProviderRuntime


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

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **kwargs: Any,
    ) -> Iterator[ProviderEvent]:
        chat_messages = to_chat_messages(messages)
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
        self.metrics["previous_response_id"] = None

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
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
        params = self._responses_kwargs(messages, tools, stream=True)
        create = cast(Any, self.client.responses.create)
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
        response_id = getattr(response, "id", None)
        if not response_id:
            return
        self.previous_response_id = str(response_id)
        self.metrics["previous_response_id"] = self.previous_response_id
        self._last_sent_message_index = self._pending_sent_message_index

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

        converted = to_responses_input(raw_input_messages)

        if self.previous_response_id is not None:
            converted = [
                item
                for item in converted
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

        input_details = getattr(usage, "input_tokens_details", None)
        cached = getattr(input_details, "cached_tokens", 0) if input_details else 0
        self.metrics["cached_tokens"] = cached or 0

        output_details = getattr(usage, "output_tokens_details", None)
        reasoning = (
            getattr(output_details, "reasoning_tokens", 0) if output_details else 0
        )
        self.metrics["reasoning_tokens"] = reasoning or 0
