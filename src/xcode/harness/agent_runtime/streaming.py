"""Provider 交互与工具执行。

从 structured.py 提取的模型调用重试、fallback、流式响应组装
和工具执行逻辑，作为 StructuredAgent 的协作模块。
"""

from __future__ import annotations

import asyncio
from typing import Any

from xcode.ai.events import (
    ProviderEvent,
    ToolCall as ToolUseBlock,
)
from xcode.ai.providers.protocol import ModelProvider
from xcode.agent.provider_retry import call_provider_with_retry
from .agent_helpers import typed_blocks_to_raw
from .cancellation import CancellationToken
from .tool_events import ToolResult
from .tool_executor import (
    ExecutionCancelled,
    ToolExecutor,
)
from ..adapters.tool_schema import tool_definitions_from_specs
from ..config import ExecutionMode
from ..observability import HookManager, PermissionPolicy
from ..skills import ApprovalCallback, ToolSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .structured import StructuredAgentEvent


# ── provider 重试 + fallback ──


async def call_provider_with_retry_harness(
    provider: ModelProvider,
    fallback_provider: ModelProvider | None,
    messages: list[dict[str, Any]],
    registry: tuple[ToolSpec, ...],
    consecutive_errors: int,
) -> tuple[list[ProviderEvent], int, bool]:
    """调用 provider，含指数退避和 fallback。返回 (events, new_consecutive_errors, switched_to_fallback)。

    委托给 agent/ 层的 call_provider_with_retry。
    """
    if provider is None:
        from xcode.ai.events import FinalMessage

        return [FinalMessage("StructuredAgent requires a provider", "error")], 0, False

    tool_definitions = tool_definitions_from_specs(registry)

    events, active_provider, new_consecutive_errors = await call_provider_with_retry(
        provider,
        messages,
        tool_definitions,
        fallback_provider=fallback_provider,
        max_retries=3,
        backoff_base=0.5,
        error_threshold=3,
        consecutive_errors=consecutive_errors,
    )

    switched = active_provider is not provider
    return events, new_consecutive_errors, switched


# ── 流式响应组装 ──


async def call_model_streaming(
    provider: ModelProvider,
    fallback_provider: ModelProvider | None,
    messages: list[dict[str, Any]],
    registry: tuple[ToolSpec, ...],
    consecutive_errors: int,
    step: int,
) -> tuple[
    list[dict[str, Any]],
    str | None,
    list[StructuredAgentEvent],
    str | None,
    int,
    bool,
]:
    """调用模型并组装为 raw blocks + stop_reason + stream events。返回 (blocks, stop_reason, events, reasoning, new_errors, switched_to_fallback)。"""
    events, new_errors, switched = await call_provider_with_retry_harness(
        provider, fallback_provider, messages, registry, consecutive_errors
    )

    from ...agent.provider_response import provider_events_to_response

    response = provider_events_to_response(events)

    from .structured import StructuredAgentEvent

    stream_events = [
        StructuredAgentEvent(f"{delta.kind}_delta", step, delta.chunk)
        for delta in response.deltas
    ]
    return (
        typed_blocks_to_raw(response.content),
        response.stop_reason,
        stream_events,
        response.reasoning_content,
        new_errors,
        switched,
    )


# ── 工具执行 ──


async def execute_tool_uses(
    *,
    uses: list[ToolUseBlock],
    registry: tuple[ToolSpec, ...],
    tool_workers: int,
    approval_callback: ApprovalCallback | None,
    permission_policy: PermissionPolicy | None,
    hook_manager: HookManager | None,
    audit_logger: Any,
    session_id: str,
    policy: Any,
    cancellation_token: CancellationToken,
    active_tool_map: dict[str, ToolSpec],
    mode: ExecutionMode,
) -> list[ToolResult]:
    cancel = asyncio.Event()
    if cancellation_token.is_cancelled():
        cancel.set()
    executor = ToolExecutor(
        registry,
        tool_workers=tool_workers,
        approval_callback=approval_callback,
        permission_policy=permission_policy,
        hook_manager=hook_manager,
        audit_logger=audit_logger,
        session_id=session_id,
        policy=policy,
    )
    try:
        return await executor.execute(
            uses,
            cancel=cancel,
            active_tool_map=active_tool_map,
            mode=mode,
        )
    except ExecutionCancelled:
        return [
            ToolResult(tool_use.id, cancellation_token.reason, "interrupted")
            for tool_use in uses
        ]
