"""Provider 重试与 fallback 逻辑。

纯 agent-core 模块，只依赖 ai/ 层。提供指数退避重试和 provider
fallback 切换能力。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from xcode.ai.events import FinalMessage, ProviderEvent
from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import ToolDefinition


def _is_transient_error(error: Exception) -> bool:
    """判断是否为瞬态错误（rate limit、overload 等）。"""
    name = type(error).__name__.lower()
    msg = str(error).lower()
    return (
        "ratelimit" in name
        or "429" in msg
        or "overloaded" in name
        or "529" in msg
        or "overloaded" in msg
    )


async def call_provider_with_retry(
    provider: ModelProvider,
    messages: list[dict],
    tools: list[ToolDefinition],
    *,
    fallback_provider: ModelProvider | None = None,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    error_threshold: int = 3,
    consecutive_errors: int = 0,
) -> tuple[list[ProviderEvent], ModelProvider, int]:
    """调用 provider，含指数退避重试和 fallback 切换。

    重试策略：
    - 瞬态错误（rate limit、overload）最多重试 max_retries 次
    - 每次重试使用指数退避 + 随机抖动
    - 非瞬态错误立即返回，不重试
    - 当 consecutive_errors 累计达到 error_threshold 时，切换到
      fallback_provider（如果提供了的话）

    参数 consecutive_errors 是跨调用的累计错误计数。调用方在每次成功
    调用后应将其重置为 0，在失败时传回。这个设计是有意的：fallback
    切换依赖跨调用的错误累积，单次调用内部的重试不重置此计数。

    返回 (events, active_provider, new_consecutive_errors)。
    active_provider 是最终成功的 provider（可能是 fallback）。
    """
    if provider is None:
        return (
            [FinalMessage("no provider configured", "error")],
            provider,
            consecutive_errors,
        )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            events: list[ProviderEvent] = []
            async for event in provider.stream(messages, tools):
                events.append(event)
            return events, provider, 0
        except Exception as e:
            last_error = e
            if not _is_transient_error(e):
                return (
                    [FinalMessage(f"Provider error: {e}", "error")],
                    provider,
                    consecutive_errors + 1,
                )

            consecutive_errors += 1

            # 累计错误达到阈值，尝试 fallback
            if consecutive_errors >= error_threshold and fallback_provider is not None:
                try:
                    events = []
                    async for event in fallback_provider.stream(messages, tools):
                        events.append(event)
                    return events, fallback_provider, 0
                except Exception as e2:
                    return (
                        [FinalMessage(f"Fallback provider error: {e2}", "error")],
                        fallback_provider,
                        0,
                    )

            if attempt >= max_retries:
                return (
                    [FinalMessage(f"Provider repeatedly unavailable: {e}", "error")],
                    provider,
                    consecutive_errors,
                )

            # 指数退避 + 抖动
            base_delay = min(backoff_base * (2 ** attempt), 32.0)
            jitter = random.uniform(0, base_delay * 0.25)
            await asyncio.sleep(base_delay + jitter)

    return (
        [FinalMessage(f"Provider unavailable: {last_error}", "error")],
        provider,
        consecutive_errors,
    )
