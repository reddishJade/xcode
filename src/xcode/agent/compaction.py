"""Token 估算和压缩触发辅助工具。

提供基于真实 provider usage 的 token 压力判断和压缩触发逻辑。
"""

from __future__ import annotations

from typing import Any

from xcode.agent.messages import AgentMessage, AssistantMessage


def estimate_tokens_simple(text: str) -> int:
    """简单 token 估算（4 字符/token）。

    仅用于本地快速估算，不精确。
    """
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[AgentMessage]) -> int:
    """估算消息列表的 token 总数（简单版本）。"""
    total = 0
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            # Assistant 消息可能有文本和工具调用
            for block in msg.content:
                # 使用类型检查避免属性访问警告
                block_type = type(block).__name__
                if block_type == "TextContent" and hasattr(block, "text"):
                    total += estimate_tokens_simple(block.text)
                elif block_type == "ThinkingContent" and hasattr(block, "thinking"):
                    total += estimate_tokens_simple(block.thinking)
                elif block_type == "ToolCallContent" and hasattr(block, "arguments"):
                    import json
                    total += estimate_tokens_simple(
                        json.dumps(block.arguments or {}, default=str)
                    )
        elif hasattr(msg, "content"):
            # User/Tool messages
            content = msg.content
            if isinstance(content, str):
                total += estimate_tokens_simple(content)
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        total += estimate_tokens_simple(block.text)
                    elif hasattr(block, "content"):
                        total += estimate_tokens_simple(block.content)
    return total


def should_compact_token_aware(
    messages: list[AgentMessage],
    *,
    last_prompt_tokens: int | None = None,
    model_soft_threshold: int = 32000,
    compact_threshold: int = 0,
    compact_token_threshold: int = 0,
) -> bool:
    """基于真实 token 和阈值判断是否需要压缩。

    优先级：
    1. 如果有 last_prompt_tokens（上次 provider 返回的真实值），用它判断
    2. 否则回退到本地估算 + 消息数阈值

    设计原因：
    - provider 返回的 prompt_tokens 比本地估算更准确
    - 避免上下文接近 model context window 时才触发压缩
    """
    # 优先使用真实 token 判断（如果有的话，只用它判断）
    if last_prompt_tokens is not None:
        return last_prompt_tokens >= model_soft_threshold

    # 回退到阈值判断（只在没有真实 token 时才使用）
    if compact_threshold > 0 and len(messages) >= compact_threshold:
        return True

    if compact_token_threshold > 0:
        estimated = estimate_message_tokens(messages)
        return estimated >= compact_token_threshold

    return False


def extract_prompt_tokens_from_usage(usage: dict[str, Any] | None) -> int | None:
    """从 usage 字典提取 prompt_tokens。

    支持 OpenAI/DeepSeek/ChatGLM 格式。
    """
    if not usage:
        return None
    return usage.get("prompt_tokens")


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
