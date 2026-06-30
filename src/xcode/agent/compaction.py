"""Token 估算和压缩触发辅助工具。

提供基于 real provider usage 和 tiktoken 的 token 压力判断和压缩触发逻辑。
"""

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
    """基于 tiktoken cl100k_base 的 token 估算。

    tiktoken 是 pyproject.toml 声明的项目级依赖，agent 层直接使用。
    不可用时回退到字节估算（约 3 字节/token）。
    """
    try:
        encoding = _get_encoding()
        return max(1, len(encoding.encode(text)))
    except BaseException:
        return max(1, len(text.encode("utf-8")) // 3)


def estimate_message_tokens(messages: Sequence[AgentMessage]) -> int:
    """估算消息列表的 token 总数。"""
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
    model_soft_threshold: int = 32000,
    compact_threshold: int = 0,
    compact_token_threshold: int = 0,
    model_name: str | None = None,
    reserve_tokens: int = 0,
) -> bool:
    """基于真实 token 和阈值判断是否需要压缩。

    优先级：
    1. 如果有 last_prompt_tokens（上次 provider 返回的真实值），用它判断
       当同时提供 model_name + reserve_tokens 时，
       使用 context_window - reserve_tokens 作为精确触发线
    2. 否则回退到本地估算 + 消息数阈值

    设计原因：
    - provider 返回的 prompt_tokens 比本地估算更准确
    - 避免上下文接近 model context window 时才触发压缩
    """
    # 优先使用真实 token 判断（如果有的话，只用它判断）
    if last_prompt_tokens is not None:
        if model_name and reserve_tokens > 0:
            trigger = effective_compact_threshold(
                model_name,
                reserve_tokens=reserve_tokens,
                fallback_threshold=model_soft_threshold,
            )
            return last_prompt_tokens >= trigger
        return last_prompt_tokens >= model_soft_threshold

    # 回退到阈值判断（只在没有真实 token 时才使用）
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


def get_model_soft_threshold(model: str | None) -> int:
    """获取模型的软 token 阈值。

    在接近上下文窗口前触发压缩，留出安全余量。
    """
    if not model:
        return 32000

    model_lower = model.lower()

    # DeepSeek 系列
    if "deepseek" in model_lower:
        if "v4" in model_lower or "v3" in model_lower:
            return 60000  # 128k context, 触发阈值约 60k
        return 28000  # 旧版本 64k context

    # ChatGLM 系列
    if "glm" in model_lower:
        if "4" in model_lower or "5" in model_lower:
            return 120000  # 256k context
        return 28000

    # MiMo 系列
    if "mimo" in model_lower:
        return 60000  # 128k context

    # OpenAI 系列
    if "gpt-4" in model_lower:
        return 120000  # 128k context
    if "gpt-3.5" in model_lower:
        return 14000  # 16k context

    # 默认保守值
    return 32000


# 模型上下文窗口映射（完整窗口大小，非阈值）
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # DeepSeek
    "deepseek-v4": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-v2": 64_000,
    # ChatGLM
    "glm-4": 256_000,
    "glm-5": 256_000,
    # MiMo
    "mimo": 128_000,
    # OpenAI
    "gpt-4": 128_000,
    "gpt-3.5": 16_000,
}


def get_model_context_window(model: str | None) -> int | None:
    """获取模型完整的上下文窗口（token 数）。

    在 soft threshold 设计不满足需求时，使用此函数获取窗口大小，
    再减去 reserve_tokens 得到精确的触发线。
    返回 None 表示未知模型，由调用方使用 fallback 阈值。
    """
    if not model:
        return None
    model_lower = model.lower()
    for prefix, window in _MODEL_CONTEXT_WINDOWS.items():
        if prefix in model_lower:
            return window
    return None


def effective_compact_threshold(
    model: str | None,
    reserve_tokens: int = 0,
    fallback_threshold: int = 32000,
) -> int:
    """计算有效压缩触发阈值。

    优先使用 model 的上下文窗口减去 reserve_tokens，
    未知模型回退到 fallback_threshold。
    """
    context_window = get_model_context_window(model)
    if context_window is not None and reserve_tokens > 0:
        return context_window - reserve_tokens
    return fallback_threshold
