"""Token 估算和压缩触发判断。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import tiktoken

from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ThinkingContent, ToolCallContent

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}
_DEFAULT_ENCODING = "cl100k_base"


def _get_encoding(name: str = _DEFAULT_ENCODING) -> tiktoken.Encoding:
    if name not in _ENCODING_CACHE:
        _ENCODING_CACHE[name] = tiktoken.get_encoding(name)
    return _ENCODING_CACHE[name]


def estimate_tokens(text: str) -> int:
    try:
        encoding = _get_encoding()
        return max(1, len(encoding.encode(text)))
    except BaseException:
        return max(1, len(text.encode("utf-8")) // 3)


def estimate_message_tokens(messages: Sequence[AgentMessage]) -> int:
    total = 0
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextContent):
                    total += estimate_tokens(block.text)
                elif isinstance(block, ThinkingContent):
                    total += estimate_tokens(block.thinking)
                elif isinstance(block, ToolCallContent):
                    total += estimate_tokens(
                        json.dumps(block.arguments or {}, default=str)
                    )
        elif isinstance(msg, (SystemMessage, UserMessage, ToolResultMessage)):
            content = msg.content
            if isinstance(content, str):
                total += estimate_tokens(content)
            else:
                for block in content:
                    if isinstance(block, TextContent):
                        total += estimate_tokens(block.text)
    return total


def should_compact_token_aware(
    messages: Sequence[AgentMessage],
    *,
    last_prompt_tokens: int | None = None,
    fallback_threshold: int = 32000,
    compact_threshold: int = 0,
    compact_token_threshold: int = 0,
    reserve_tokens: int = 16384,
) -> bool:
    if last_prompt_tokens is not None:
        trigger = fallback_threshold
        from xcode.ai.models import get_model_context_window

        window = get_model_context_window(model_name=None) if False else None
        if window is not None and reserve_tokens > 0:
            trigger = window - reserve_tokens
        return last_prompt_tokens >= trigger

    if compact_threshold > 0 and len(messages) >= compact_threshold:
        return True

    if compact_token_threshold > 0:
        estimated = estimate_message_tokens(messages)
        return estimated >= compact_token_threshold

    return False


def extract_prompt_tokens_from_usage(usage: Mapping[str, object] | None) -> int | None:
    if not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
        return prompt_tokens
    return None
