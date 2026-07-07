"""智谱 AI ChatGLM provider（兼容 OpenAI Chat API）。"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any, cast

from xcode.ai.events import ProviderEvent
from xcode.ai.types import ProviderConfig, ToolDefinition

from ._codec import to_chat_messages, to_chat_tools
from ._compat import OpenAICompatProvider

CHATGLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ChatGLMProvider(OpenAICompatProvider):
    """智谱 AI ChatGLM API 适配。"""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(config, transport="chatglm_chat", client=client)
        self._metrics["prompt_tokens"] = 0
        self._metrics["completion_tokens"] = 0
        self._metrics["total_tokens"] = 0

    @property
    def _clear_thinking(self) -> bool:
        return self.config.extra.get("clear_thinking", False)

    @property
    def _tool_stream(self) -> bool:
        return self.config.extra.get("tool_stream", True)

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        thinking: bool | None = None,
    ) -> Iterator[ProviderEvent]:
        params = self._chat_kwargs(
            messages,
            tools,
            stream=True,
            thinking=thinking,
        )
        openai_messages = cast(list[dict[str, Any]], params["messages"])
        yield from self._call_chat_api(params, len(openai_messages))

    def _chat_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        *,
        stream: bool,
        thinking: bool | None = None,
    ) -> dict[str, object]:
        cleaned_messages = self._clean_reasoning_content(messages)
        openai_messages = to_chat_messages(cleaned_messages)
        effective_thinking = (
            self.config.thinking if thinking is None else thinking
        )

        kwargs: dict[str, object] = {
            "model": self.config.model,
            "messages": openai_messages,
            "tools": to_chat_tools(tools),
            "stream": stream,
        }
        if self.config.response_format:
            kwargs["response_format"] = self.config.response_format

        self._build_thinking_params(kwargs, effective_thinking)

        if stream and self._tool_stream and _supports_tool_stream(self.config.model):
            extra_body = cast(
                dict[str, Any], kwargs.setdefault("extra_body", {})
            )
            extra_body["tool_stream"] = True
        extra_body = cast(dict[str, Any], kwargs.setdefault("extra_body", {}))
        thinking_body = cast(
            dict[str, Any], extra_body.setdefault("thinking", {})
        )
        thinking_body["clear_thinking"] = self._clear_thinking
        return kwargs

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not messages or not self._clear_thinking:
            return messages
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            msg.pop("reasoning_content", None)
        return cleaned

    def _record_usage(self, response, sent_messages: int) -> None:
        super()._record_usage(response, sent_messages)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self._metrics["prompt_tokens"] = prompt_tokens
            self._metrics["completion_tokens"] = completion_tokens
            self._metrics["total_tokens"] = (
                getattr(usage, "total_tokens", 0)
                or prompt_tokens + completion_tokens
            )
            cached = self._metrics.get("cached_tokens", 0)
            if isinstance(cached, int) and cached > 0:
                self._metrics["cache_hit_tokens"] = cached
                self._metrics["cache_miss_tokens"] = self._metrics.get(
                    "prompt_cache_miss_tokens", 0
                )


def _supports_tool_stream(model: str) -> bool:
    return model.startswith("glm-4.6") or model.startswith("glm-4.7")
