from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from xcode.ai.types import ToolDefinition

from .codec import to_chat_messages, to_chat_tool
from .openai_compat import OpenAICompatProvider

"""DeepSeek provider（兼容 OpenAI Chat API，带 reasoning_content 支持）。"""

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAICompatProvider):
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
        runtime=None,
        strict_tools: bool = False,
        response_format: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            runtime=runtime,
            transport="deepseek_chat",
            client=client,
        )
        self.strict_tools = strict_tools
        self.response_format = response_format
        self.metrics["prompt_cache_hit_tokens"] = 0
        self.metrics["prompt_cache_miss_tokens"] = 0

    def _stream_sync(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        if self.thinking:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("presence_penalty", None)
            kwargs.pop("frequency_penalty", None)

        cleaned_messages = self._clean_reasoning_content(messages)
        api_messages = to_chat_messages(cleaned_messages)

        # 优先使用运行时传递的 response_format，回退到构造时配置
        effective_format = response_format or self.response_format
        if effective_format and effective_format.get("type") == "json_object":
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

        if effective_format:
            params["response_format"] = effective_format
        if max_tokens:
            params["max_tokens"] = max_tokens

        self._build_thinking_params(params)

        for k, v in kwargs.items():
            if k not in params:
                params[k] = v

        yield from self._call_chat_api(params, len(api_messages))

    def _clean_reasoning_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """清理 reasoning_content 以符合 DeepSeek API 要求。

        DeepSeek API 的 reasoning_content 处理规则：
        1. 非工具循环（最后一条不是 tool 消息）：
           - 清除所有历史 reasoning_content
           - 原因：避免累积大量思考内容导致上下文膨胀

        2. 工具循环（最后一条是 tool 消息）：
           - 保留当前轮次（最后一个 user 之后）的 reasoning_content
           - 清除之前轮次的 reasoning_content
           - 原因：DeepSeek API 要求工具调用时保留当前轮次思考，否则 API 报错

        这是 DeepSeek 特有的 API 约束，其他 provider 不需要此逻辑。
        """
        if not messages:
            return messages

        cleaned = copy.deepcopy(messages)
        in_tool_loop = cleaned[-1].get("role") == "tool"

        if not in_tool_loop:
            # 非工具循环：清除所有 reasoning_content
            for msg in cleaned:
                msg.pop("reasoning_content", None)
        else:
            # 工具循环：保留当前轮次，清除历史
            last_user_idx = -1
            for i in range(len(cleaned) - 1, -1, -1):
                if cleaned[i].get("role") == "user":
                    last_user_idx = i
                    break
            # 清除最后一个 user 之前的所有 reasoning_content
            for i in range(last_user_idx):
                if cleaned[i].get("role") == "assistant":
                    cleaned[i].pop("reasoning_content", None)

        return cleaned

    def _ensure_json_word(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """确保消息中包含 'json' 关键字，否则注入提示。

        设计原因：
        DeepSeek API 在 response_format={"type": "json_object"} 时要求
        prompt 中必须包含 'json' 关键字，否则 API 返回 400 错误。
        这是 DeepSeek 特有的约束（OpenAI API 没有此要求）。

        注入策略：
        1. 优先追加到 system 消息
        2. 若无 system 消息，追加到第一条消息
        """
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
        """记录 DeepSeek usage，优先使用原生缓存字段。"""
        from xcode.ai.cache import extract_cache_usage

        self.metrics["sent_messages"] = sent_messages
        usage = getattr(response, "usage", None)
        if usage:
            # 使用统一的缓存提取逻辑
            cache_usage = extract_cache_usage(response)
            self.metrics["prompt_cache_hit_tokens"] = cache_usage.hit_tokens
            self.metrics["prompt_cache_miss_tokens"] = cache_usage.miss_tokens
            self.metrics["cached_tokens"] = cache_usage.hit_tokens
            self.metrics["cache_hit_rate"] = cache_usage.hit_rate

            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                getattr(completion_details, "reasoning_tokens", 0)
                if completion_details
                else 0
            )
            self.metrics["reasoning_tokens"] = reasoning or 0
