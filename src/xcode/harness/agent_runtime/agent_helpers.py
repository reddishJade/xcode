"""StructuredAgent 工具函数。

从 structured.py 提取的纯函数和辅助逻辑：block 转换、指标聚合、
watchdog、sync/async 桥接、provider 事件收集。
"""

from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

from ...agent.messages import convert_to_llm
from ...agent.types import (
    AgentMessage,
    ContentBlock,
    TextContent,
    ToolCallBlock,
)
from xcode.ai.events import (
    ProviderEvent,
    ToolCall as ToolUseBlock,
)
from xcode.ai.providers.protocol import ModelProvider
from xcode.harness.adapters.tool_schema import tool_definitions_from_specs
from .cancellation import CancellationToken
from .compaction import (
    budget_large_tool_outputs,
    estimate_message_tokens,
    estimate_text_tokens,
    latest_read_file_tool_result_ids,
)
from .tool_executor import stringify_tool_input
from ..skills import ToolSpec
from .async_worker import IsolatedAsyncWorker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .structured import StructuredAgentEvent


# ── 消息/块 转换 ──


def to_dict(msg: AgentMessage) -> dict[str, Any]:
    """将类型化消息转为 dict（保持 state.messages 的 dict 格式）。"""
    result = convert_to_llm([msg])
    assert result, f"convert_to_llm returned empty for {type(msg).__name__}"
    return result[0]


def blocks_to_typed(blocks: list[dict[str, Any]]) -> list[ContentBlock]:
    """将 raw dict content blocks 转为类型化 ContentBlock。"""
    result: list[ContentBlock] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            result.append(TextContent(text=str(b.get("text", ""))))
        elif b.get("type") == "tool_use":
            result.append(
                ToolCallBlock(
                    id=str(b.get("id", "")),
                    name=str(b.get("name", "")),
                    arguments=b.get("input", {}),
                )
            )
    return result


def typed_blocks_to_raw(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """将 Agent ContentBlock 转为 StructuredAgent 运行状态使用的 raw block。"""
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallBlock):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments or {},
                }
            )
    return result


# ── block 查询 ──


def is_tool_use(block: dict[str, Any]) -> bool:
    return block.get("type") == "tool_use"


def to_tool_use(block: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(
        id=str(block.get("id", "")),
        name=str(block.get("name", "")),
        input=block.get("input", {}),
    )


def text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif "text" in block:
            parts.append(str(block["text"]))
    return "".join(parts).strip()


def reasoning_for_assistant(
    blocks: list[dict[str, Any]],
    reasoning_content: str | None,
) -> str | None:
    if reasoning_content is not None:
        return reasoning_content
    if any(is_tool_use(block) for block in blocks):
        return ""
    return None


# ── 预算/指标 ──


def budget_messages_for_provider(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """发送模型前裁剪过大的非文件读取工具结果。"""
    preserved_tool_results = latest_read_file_tool_result_ids(messages)
    return budget_large_tool_outputs(
        messages,
        large_tool_output_chars=8_000,
        large_tool_output_head_chars=4_000,
        large_tool_output_tail_chars=4_000,
        compact_token_threshold=1,
        budget_trigger_token_ratio=0,
        preserve_tool_result_ids=preserved_tool_results,
    )


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(metrics)
    finalized["model_total_ms"] = round(sum(metrics["model_latencies_ms"]), 3)
    finalized["tool_total_ms"] = round(sum(metrics["tool_latencies_ms"]), 3)
    finalized["total_observed_ms"] = round(
        finalized["model_total_ms"] + finalized["tool_total_ms"],
        3,
    )
    return finalized


# ── watchdog ──


def check_repeated_tool_watchdog(
    uses: list[ToolUseBlock],
    last_signature: str | None,
    repeated_count: int,
    limit: int,
) -> tuple[str | None, str | None, int]:
    if limit <= 0 or len(uses) != 1:
        return None, None, 0
    signature = f"{uses[0].name}:{stringify_tool_input(uses[0].input)}"
    if signature == last_signature:
        repeated_count += 1
    else:
        repeated_count = 1
    if repeated_count > limit:
        return (
            f"watchdog stopped repeated tool call: {uses[0].name}",
            signature,
            repeated_count,
        )
    return None, signature, repeated_count


# ── sync/async 桥接 ──


def run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if hasattr(coro, "close"):
        coro.close()
    raise RuntimeError(
        "StructuredAgent.run() cannot be called inside an active event loop; "
        "use await StructuredAgent.run_async() instead."
    )


def aiter_to_sync_iter(
    async_iter: AsyncIterator[StructuredAgentEvent],
    cancellation_token: CancellationToken,
) -> Iterator[StructuredAgentEvent]:
    items: queue.Queue[tuple[str, Any]] = queue.Queue()
    worker = IsolatedAsyncWorker(name="xcode-sync-stream-worker")

    async def consume() -> None:
        try:
            async for event in async_iter:
                items.put(("item", event))
        except BaseException as exc:
            items.put(("error", exc))
        finally:
            items.put(("done", None))

    future = worker.submit(consume())
    try:
        while True:
            kind, payload = items.get()
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                return
    finally:
        if not future.done():
            cancellation_token.cancel("sync stream consumer stopped")
            future.cancel()
        worker.close()


# ── provider 事件收集 ──


async def collect_provider_events(
    provider: ModelProvider,
    messages: list[dict[str, Any]],
    registry: tuple[ToolSpec, ...],
) -> list[ProviderEvent]:
    events = []
    tool_definitions = tool_definitions_from_specs(registry)
    async for event in provider.stream(messages, tool_definitions):
        events.append(event)
    return events
