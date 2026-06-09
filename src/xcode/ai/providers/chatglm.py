from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from xcode.ai.events import ProviderEvent
from xcode.ai.types import StreamOptions, ToolDefinition

from .codec import to_chat_tool
from .openai_compat import OpenAICompatProvider

"""智谱 AI ChatGLM provider（兼容 OpenAI Chat API）。

支持模型：glm-4、glm-4-flash、glm-4.7、glm-5、glm-5.1 等。
GLM-4.7+ 默认开启 thinking，支持交错式思考和保留式思考。
API 文档：https://docs.bigmodel.cn/
"""

CHATGLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class ChatGLMProvider(OpenAICompatProvider):
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
        runtime=None,
        client=None,
    ) -> None:
        super().__init__(
            api_key,
            base_url or CHATGLM_BASE_URL,
            model,
            thinking=thinking,
            reasoning_effort=None,
            runtime=runtime,
            client=client,
            transport="chatglm_chat",
            import_error_msg="Missing dependency: openai.",
        )
        self.clear_thinking = clear_thinking
        self.tool_stream = tool_stream
        self.response_format = response_format
        self.base_url = base_url or CHATGLM_BASE_URL
        self.metrics["prompt_tokens"] = 0
        self.metrics["completion_tokens"] = 0
        self.metrics["total_tokens"] = 0

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        *,
        response_format: dict[str, Any] | None = None,
        thinking: bool | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        self._current_options = options
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
        **kwargs: Any,
    ) -> Iterator[ProviderEvent]:
        clean = self._clean_reasoning_content(messages)
        from .codec import to_chat_messages
        api_messages = to_chat_messages(clean)
        params = self._build_chat_params(
            api_messages, tools, response_format=response_format, thinking=thinking, **kwargs
        )
        yield from self._call_chat_api(params, len(api_messages))

    def _build_chat_params(
        self,
        api_messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        **kwargs: Any,
    ) -> dict[str, Any]:
        thinking_override = kwargs.pop("thinking", None)
        effective_thinking = self.thinking if thinking_override is None else thinking_override
        effective_format = kwargs.pop("response_format", None) or self.response_format

        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "tools": [to_chat_tool(t.name, t.description, t.schema) for t in tools],
            "stream": True,
        }
        if effective_format:
            params["response_format"] = effective_format
        if self.tool_stream and _supports_tool_stream(self.model):
            params["tool_stream"] = True
        self._build_thinking_params(params, effective_thinking)
        extra_body = cast(dict[str, Any], params.setdefault("extra_body", {}))
        thinking_body = cast(dict[str, Any], extra_body.setdefault("thinking", {}))
        thinking_body["clear_thinking"] = self.clear_thinking
        return params

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages
        if not self.clear_thinking:
            return copy.deepcopy(messages)

        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            msg.pop("reasoning_content", None)

        return cleaned

    def _record_usage(self, response, sent_messages: int) -> None:
        """记录 ChatGLM usage，添加 provider 专属 token 统计。"""
        super()._record_usage(response, sent_messages)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(
                usage, "total_tokens", prompt_tokens + completion_tokens
            )
            self.metrics["prompt_tokens"] = prompt_tokens
            self.metrics["completion_tokens"] = completion_tokens
            self.metrics["total_tokens"] = total_tokens or 0


def _supports_tool_stream(model: str) -> bool:
    """判断模型是否支持 tool_stream。

    tool_stream 支持版本（来源：ChatGLM API 文档）：
    - glm-4.6: 支持
    - glm-4.7: 支持
    - 更早版本: 不支持
    - 未来版本: 需要查阅官方文档确认
    """
    return model.startswith("glm-4.6") or model.startswith("glm-4.7")
