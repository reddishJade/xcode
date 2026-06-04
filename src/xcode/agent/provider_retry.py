"""Provider 重试逻辑。

纯 agent-core 模块，只依赖 ai/ 层。提供指数退避重试能力。
"""

from __future__ import annotations

import asyncio
import random
from xcode.ai.events import FinalMessage, ProviderEvent
from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import StreamOptions, ToolDefinition


def _is_transient_error(error: Exception) -> bool:
    """判断是否为瞬态错误（rate limit、overload 等）。

    使用通用的 HTTP 状态码和标准错误名称检测，适用于所有 OpenAI API 兼容 provider。
    """
    name = type(error).__name__.lower()
    msg = str(error).lower()
    return (
        "ratelimit" in name
        or "rate_limit" in name
        or "429" in msg
        or "overloaded" in msg
        or "529" in msg
        or "503" in msg
        or "502" in msg
    )


async def call_provider_with_retry(
    provider: ModelProvider,
    messages: list[dict],
    tools: list[ToolDefinition],
    *,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    options: StreamOptions | None = None,
) -> list[ProviderEvent]:
    """调用 provider，含指数退避重试。

    重试策略：
    - 瞬态错误（rate limit、overload）最多重试 max_retries 次
    - 每次重试使用指数退避 + 随机抖动
    - 非瞬态错误立即返回，不重试

    参数:
        provider: ModelProvider 实例
        messages: LLM 格式消息列表
        tools: 工具定义列表
        max_retries: 最大重试次数
        backoff_base: 基础退避时间（秒）
        options: 每请求选项（api_key, session_id 等）

    返回:
        ProviderEvent 列表
    """
    if provider is None:
        return [FinalMessage("no provider configured", "error")]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            events: list[ProviderEvent] = []
            stream_kwargs = {}
            if options is not None:
                stream_kwargs["options"] = options
            async for event in provider.stream(messages, tools, **stream_kwargs):
                events.append(event)
            return events
        except Exception as e:
            last_error = e
            if not _is_transient_error(e):
                return [FinalMessage(f"Provider error: {e}", "error")]

            if attempt >= max_retries:
                return [FinalMessage(f"Provider repeatedly unavailable: {e}", "error")]

            # 指数退避 + 抖动
            base_delay = min(backoff_base * (2**attempt), 32.0)
            jitter = random.uniform(0, base_delay * 0.25)
            await asyncio.sleep(base_delay + jitter)

    return [FinalMessage(f"Provider unavailable: {last_error}", "error")]
