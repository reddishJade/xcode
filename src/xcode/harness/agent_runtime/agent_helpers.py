"""StructuredAgent 工具函数。

从 structured.py 提取的纯函数和辅助逻辑：消息转换、预算裁剪、sync/async 桥接。
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from ...agent.message_converter import convert_to_llm
from ...agent.messages import AgentMessage, ToolResultMessage
from ...agent.types import ShellCallOutputContent, TextContent
from .cancellation import CancellationToken
from .compaction import (
    budget_large_tool_outputs,
    latest_read_file_tool_result_ids,
)
from .async_worker import IsolatedAsyncWorker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import StructuredAgentEvent

T = TypeVar("T")


# ── 消息/块 转换 ──


def to_dict(msg: AgentMessage) -> dict[str, Any]:
    """将类型化消息转为 dict（保持 state.messages 的 dict 格式）。"""
    if isinstance(msg, ToolResultMessage):
        return _tool_result_message_to_dict(msg)
    result = convert_to_llm([msg])
    assert result, f"convert_to_llm returned empty for {type(msg).__name__}"
    return result[0]


def _tool_result_message_to_dict(msg: ToolResultMessage) -> dict[str, Any]:
    """将工具结果保留为状态可见的结构化记录。"""
    status = _tool_result_status(msg)
    if isinstance(msg.content, list):
        content = []
        for item in msg.content:
            if isinstance(item, TextContent):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": item.text,
                        "status": status,
                    }
                )
            elif isinstance(item, ShellCallOutputContent):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": str(item.output),
                        "status": status,
                    }
                )
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": content}
    return {
        "role": "tool",
        "tool_call_id": msg.tool_call_id,
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": str(msg.content),
                "status": status,
            }
        ],
    }


def _tool_result_status(msg: ToolResultMessage) -> str:
    """根据工具结果错误状态生成 UI 可见状态。"""
    if not msg.is_error:
        return "ok"
    text = _tool_result_text(msg.content).lower()
    if "interrupt" in text or "cancel" in text:
        return "interrupted"
    return "error"


def _tool_result_text(content: object) -> str:
    """提取工具结果文本，用于状态归类。"""
    if isinstance(content, list):
        return "".join(item.text for item in content if isinstance(item, TextContent))
    return str(content)


def text_from_blocks(blocks: list[Mapping[str, object]]) -> str:
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            text = block.get("text")
            if text is not None:
                parts.append(str(text))
        elif "text" in block:
            text = block["text"]
            if text is not None:
                parts.append(str(text))
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


def run_coro_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "StructuredAgent.run() cannot be called inside an active event loop; "
        "use await StructuredAgent.run_async() instead."
    )


@dataclass(frozen=True)
class _StreamItem:
    event: StructuredAgentEvent


@dataclass(frozen=True)
class _StreamError:
    error: BaseException


@dataclass(frozen=True)
class _StreamDone:
    pass


_StreamMessage = _StreamItem | _StreamError | _StreamDone


def aiter_to_sync_iter(
    async_iter: AsyncIterator[StructuredAgentEvent],
    cancellation_token: CancellationToken,
) -> Iterator[StructuredAgentEvent]:
    items: queue.Queue[_StreamMessage] = queue.Queue()
    worker = IsolatedAsyncWorker(name="xcode-sync-stream-worker")

    async def consume() -> None:
        try:
            async for event in async_iter:
                items.put(_StreamItem(event))
        except BaseException as exc:
            items.put(_StreamError(exc))
        finally:
            items.put(_StreamDone())

    future = worker.submit(consume())
    try:
        while True:
            message = items.get()
            if isinstance(message, _StreamItem):
                yield message.event
            elif isinstance(message, _StreamError):
                raise message.error
            elif isinstance(message, _StreamDone):
                return
    finally:
        if not future.done():
            cancellation_token.cancel("sync stream consumer stopped")
            future.cancel()
        worker.close()
