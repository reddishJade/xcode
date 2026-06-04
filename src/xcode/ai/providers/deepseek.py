from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from .codec import to_chat_messages, to_chat_tool
from .metrics import ProviderMetricsMixin
from .stream_codec import chat_stream_to_events
from .runtime import ProviderRuntime

"""DeepSeek provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。"""

# DeepSeek API 地址
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(ProviderMetricsMixin):
    """DeepSeek Chat API 适配。

    和 OpenAIChatProvider 基本一致，额外处理 reasoning_content 字段。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = "deepseek-v4-pro",
        thinking: bool = True,
        reasoning_effort: str | None = "high",
        runtime: ProviderRuntime | None = None,
        client=None,
        strict_tools: bool = False,
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
        self.strict_tools = strict_tools
        self.transport = "deepseek_chat"
        self._ensure_metrics()
        self.metrics["prompt_cache_hit_tokens"] = 0
        self.metrics["prompt_cache_miss_tokens"] = 0

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        for event in self._stream_sync(
            messages, tuple(tools), response_format, max_tokens, **kwargs
        ):
            yield event

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        if self.thinking:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("presence_penalty", None)
            kwargs.pop("frequency_penalty", None)

        # Handle reasoning_content cleanup for tool loop and turns
        cleaned_messages = self._clean_reasoning_content(messages)
        api_messages = to_chat_messages(cleaned_messages)

        if response_format and response_format.get("type") == "json_object":
            api_messages = self._ensure_json_word(api_messages)

        strict_tools = kwargs.pop("strict_tools", getattr(self, "strict_tools", False))

        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "tools": [
                to_chat_tool(
                    getattr(t, "name", ""),
                    getattr(t, "description", ""),
                    getattr(t, "schema", None),
                    strict=strict_tools,
                )
                for t in tools
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if response_format:
            params["response_format"] = response_format
        if max_tokens:
            params["max_tokens"] = max_tokens

        extra_body = {}
        if not self.thinking:
            extra_body["thinking"] = {"type": "disabled"}
        elif self.reasoning_effort:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            params["extra_body"] = extra_body
        if self.thinking and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        for k, v in kwargs.items():
            if k not in params:
                params[k] = v

        create = cast(Any, self.client.chat.completions.create)
        stream = self.runtime.run(lambda: create(**params))
        self._ensure_metrics()
        self.metrics["sent_messages"] = len(params["messages"])
        yield from chat_stream_to_events(self._intercept_usage(stream, len(params["messages"])))

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages

        cleaned = copy.deepcopy(messages)
        in_tool_loop = cleaned[-1].get("role") == "tool"

        # Boundary condition handling:
        # 1. When in_tool_loop is False:
        #    This happens either at the start of a completely new session OR when the tool loop finishes,
        #    the model returns the final answer, and then the user sends a new turn message (the last message
        #    role becomes 'user'). In both cases, we strip all reasoning_content from past messages to
        #    clean up the context and prevent API 400s or token wastes on historical completed thoughts.
        # 2. When in_tool_loop is True:
        #    This is inside an active tool loop turn. We only keep the reasoning_content for assistant
        #    messages in the current loop (after the last user message) to fulfill DeepSeek API requirements.
        #    All assistant thoughts prior to the last user message are stripped.
        if not in_tool_loop:
            for msg in cleaned:
                msg.pop("reasoning_content", None)
        else:
            last_user_idx = -1
            for i in range(len(cleaned) - 1, -1, -1):
                if cleaned[i].get("role") == "user":
                    last_user_idx = i
                    break
            for i in range(last_user_idx):
                if cleaned[i].get("role") == "assistant":
                    cleaned[i].pop("reasoning_content", None)

        return cleaned

    def _ensure_json_word(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        has_json_word = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and "json" in content.lower():
                has_json_word = True
                break
            elif isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and "json" in part.get("text", "").lower()
                    ):
                        has_json_word = True
                        break
        if not has_json_word:
            messages = copy.deepcopy(messages)
            appended = False
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content")
                    if isinstance(content, str):
                        msg["content"] = (
                            content + "\nNote: Output must be in JSON format."
                        )
                        appended = True
                        break
            if not appended and messages:
                first_msg = messages[0]
                content = first_msg.get("content")
                if isinstance(content, str):
                    first_msg["content"] = (
                        content + "\nNote: Output must be in JSON format."
                    )
                elif isinstance(content, list):
                    content.append(
                        {"type": "text", "text": "Note: Output must be in JSON format."}
                    )
        return messages

    def _record_usage(self, response, sent_messages: int) -> None:
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
            if not hit:
                details = getattr(usage, "prompt_tokens_details", None)
                if details:
                    hit = getattr(details, "cached_tokens", 0) or 0
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    if not miss and prompt_tokens:
                        miss = prompt_tokens - hit
            self.metrics["prompt_cache_hit_tokens"] = hit
            self.metrics["prompt_cache_miss_tokens"] = miss
            self.metrics["cached_tokens"] = hit

            # reasoning_tokens
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0
