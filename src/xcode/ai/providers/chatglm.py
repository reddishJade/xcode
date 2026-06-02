from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.types import ToolDefinition
from .codec import chat_stream_to_events, to_chat_messages, to_chat_tool
from .runtime import ProviderRuntime

"""智谱 AI ChatGLM provider（兼容 OpenAI Chat API）。

支持模型：glm-4、glm-4-flash、glm-4.7、glm-5、glm-5.1 等。
GLM-4.7+ 默认开启 thinking，支持交错式思考和保留式思考。
API 文档：https://docs.bigmodel.cn/
"""

# 智谱 AI OpenAI 兼容 API 地址
CHATGLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ChatGLMProvider:
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
        self.reasoning_effort = None
        self.runtime = runtime or ProviderRuntime()
        self.transport = "chatglm_chat"
        self.metrics: dict[str, object] = {
            "transport": self.transport,
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
        cleaned_messages = self._clean_reasoning_content(messages)
        openai_messages = to_chat_messages(cleaned_messages)

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": openai_messages,
            "tools": [to_chat_tool(t.name, t.description, t.schema) for t in tools],
            "stream": True,
        }

        # 工具流式输出（仅 glm-4.6/4.7 支持）
        if self.tool_stream and _supports_tool_stream(self.model):
            kwargs["tool_stream"] = True

        # thinking 配置
        extra_body: dict[str, Any] = {}
        if not self.thinking:
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {
                "type": "enabled",
                "clear_thinking": self.clear_thinking,
            }
        if extra_body:
            kwargs["extra_body"] = extra_body

        def intercept_usage(chunks):
            for chunk in chunks:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._record_usage(chunk, len(openai_messages))
                yield chunk

        stream = self.runtime.run(lambda: self.client.chat.completions.create(**kwargs))
        self.metrics["sent_messages"] = len(openai_messages)
        yield from chat_stream_to_events(intercept_usage(stream))

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
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            self.metrics["cached_tokens"] = cached or 0

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0


def _supports_tool_stream(model: str) -> bool:
    return model.startswith("glm-4.6") or model.startswith("glm-4.7")
