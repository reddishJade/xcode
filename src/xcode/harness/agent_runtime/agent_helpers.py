"""StructuredAgent 工具函数。

从 structured.py 提取的纯函数和辅助逻辑：消息转换、预算裁剪、sync/async 桥接。
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ...agent.messages import convert_to_llm
from ...agent.types import AgentMessage
from .cancellation import CancellationToken
from .compaction import (
    budget_large_tool_outputs,
    latest_read_file_tool_result_ids,
)
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


def text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif "text" in block:
            parts.append(str(block["text"]))
    return "".join(parts).strip()


# ── 预算 ──


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