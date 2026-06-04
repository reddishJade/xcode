from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from xcode.ai.events import ProviderEvent
from xcode.ai.types import ToolDefinition
from .codec import to_chat_messages, to_chat_tool
from .metrics import ProviderMetricsMixin
from .stream_codec import chat_stream_to_events
from .runtime import ProviderRuntime

"""智谱 AI ChatGLM provider（兼容 OpenAI Chat API）。

支持模型：glm-4、glm-4-flash、glm-4.7、glm-5、glm-5.1 等。
GLM-4.7+ 默认开启 thinking，支持交错式思考和保留式思考。
API 文档：https://docs.bigmodel.cn/
"""

# 智谱 AI OpenAI 兼容 API 地址
CHATGLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ChatGLMProvider(ProviderMetricsMixin):
    """智谱 AI ChatGLM API 适配。

    使用 OpenAI 兼容接口，支持 thinking 模式和保留式思考。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = CHATGLM_BASE_URL,
        model: str = "glm-4-flash",
        thinking: bool = True,
        clear_thinking: bool = False,
        tool_stream: bool = True,
        response_format: dict[str, Any] | None = None,
        runtime: ProviderRuntime | None = None,
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Missing dependency: openai.") from exc
            client = OpenAI(api_key=api_key, base_url=base_url or CHATGLM_BASE_URL)
        self.client = client
        self.model = model
        self.thinking = thinking
        self.clear_thinking = clear_thinking
        self.tool_stream = tool_stream
        self.response_format = response_format
        self.reasoning_effort = None
        self.base_url = base_url or CHATGLM_BASE_URL
        self.runtime = runtime or ProviderRuntime()
        self.transport = "chatglm_chat"
        self._ensure_metrics()
        self.metrics["prompt_tokens"] = 0
        self.metrics["completion_tokens"] = 0
        self.metrics["total_tokens"] = 0
        self.metrics["cache_hit_ratio"] = 0.0

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        *,
        response_format: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        for event in self._stream_sync(
            messages,
            tuple(tools),
            response_format=response_format,
            thinking=thinking,
        ):
            yield event

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        response_format: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> Iterator[ProviderEvent]:
        kwargs = self._chat_kwargs(
            messages,
            tools,
            stream=True,
            response_format=response_format,
            thinking=thinking,
        )
        openai_messages = cast(list[dict[str, Any]], kwargs["messages"])

        create = cast(Any, self.client.chat.completions.create)
        stream = self.runtime.run(lambda: create(**kwargs))
        self.metrics["sent_messages"] = len(openai_messages)
        yield from chat_stream_to_events(self._intercept_usage(stream, len(openai_messages)))

    def _chat_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        *,
        stream: bool,
        response_format: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> dict[str, object]:
        cleaned_messages = self._clean_reasoning_content(messages)
        openai_messages = to_chat_messages(cleaned_messages)
        effective_thinking = self.thinking if thinking is None else thinking

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": openai_messages,
            "tools": [to_chat_tool(t.name, t.description, t.schema) for t in tools],
            "stream": stream,
        }
        effective_response_format = response_format or self.response_format
        if effective_response_format:
            kwargs["response_format"] = effective_response_format

        # 工具流式输出（仅 glm-4.6/4.7 支持）
        if stream and self.tool_stream and _supports_tool_stream(self.model):
            kwargs["tool_stream"] = True

        # thinking 配置
        extra_body: dict[str, Any] = {}
        if not effective_thinking:
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {
                "type": "enabled",
                "clear_thinking": self.clear_thinking,
            }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """按 clear_thinking 策略处理 reasoning_content。

        保留式思考需要原样返回历史 reasoning_content。
        """
        if not messages:
            return messages
        if not self.clear_thinking:
            return copy.deepcopy(messages)

        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            msg.pop("reasoning_content", None)

        return cleaned

    def _record_usage(self, response, sent_messages: int) -> None:
        """记录 usage 指标，包含缓存 Token 和 reasoning_tokens 统计。"""
        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(
                usage,
                "total_tokens",
                prompt_tokens + completion_tokens,
            )
            self.metrics["prompt_tokens"] = prompt_tokens
            self.metrics["completion_tokens"] = completion_tokens
            self.metrics["total_tokens"] = total_tokens or 0

            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            self.metrics["cached_tokens"] = cached or 0
            self.metrics["cache_hit_ratio"] = (
                round((cached or 0) / prompt_tokens, 4) if prompt_tokens else 0.0
            )

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0


def _supports_tool_stream(model: str) -> bool:
    return model.startswith("glm-4.6") or model.startswith("glm-4.7")
