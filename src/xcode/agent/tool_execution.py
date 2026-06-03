"""Agent 工具执行。

从 agent_loop.py 提取的工具调用执行逻辑：串行/并行调度、
单工具执行、before/after 钩子。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentToolResult,
    AssistantMessage,
    BeforeToolCallContext,
    CancellationSignal,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessage,
)


@dataclass
class ExecutedToolBatch:
    results: list[ToolResultMessage]
    terminate: bool


async def execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    """根据 execution_mode 调度串行或并行工具执行。"""
    has_sequential = False
    for tc in tool_calls:
        if isinstance(tc, ToolCallContent):
            for t in current_context.tools or []:
                if t.name == tc.name and t.execution_mode == "sequential":
                    has_sequential = True
                    break
            if has_sequential:
                break

    if config.tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )
    return await _execute_parallel(
        current_context, assistant_message, tool_calls, config, signal, emit
    )


async def _execute_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    results: list[ToolResultMessage] = []
    for tc in tool_calls:
        if not isinstance(tc, ToolCallContent):
            continue
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments
            )
        )
        result = await _execute_one(
            current_context, assistant_message, tc, config, signal, emit
        )
        results.append(result)
        if _is_cancelled(signal):
            break
    return ExecutedToolBatch(results=results, terminate=False)


async def _execute_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    tasks = []
    for tc in tool_calls:
        if not isinstance(tc, ToolCallContent):
            continue
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments
            )
        )
        tasks.append(
            _execute_one(current_context, assistant_message, tc, config, signal, emit)
        )

    raw = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[ToolResultMessage] = [
        r for r in raw if isinstance(r, ToolResultMessage)
    ]
    return ExecutedToolBatch(results=results, terminate=False)


async def _execute_one(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> ToolResultMessage:
    tool = None
    for t in current_context.tools or []:
        if t.name == tool_call.name:
            tool = t
            break

    if tool is None:
        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=f"Tool {tool_call.name} not found",
            is_error=True,
        )

    args = tool_call.arguments or {}

    if _is_cancelled(signal):
        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=_cancel_reason(signal),
            is_error=True,
        )

    if config.before_tool_call:
        ctx = BeforeToolCallContext(
            assistant_message=assistant_message,
            tool_call=tool_call,
            args=args,
            context=current_context,
        )
        before_result = config.before_tool_call(ctx, signal)
        if before_result and before_result.block:
            return ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=before_result.reason or "Tool execution was blocked",
                is_error=True,
            )

    try:
        tool_result = await tool.execute(tool_call.id, args, signal)
        is_error = False
        content = tool_result.content
    except Exception as e:
        tool_result = AgentToolResult(content=[TextContent(text=f"Tool error: {e}")])
        content = tool_result.content
        is_error = True

    if config.after_tool_call:
        after_ctx = AfterToolCallContext(
            assistant_message=assistant_message,
            tool_call=tool_call,
            args=args,
            result=tool_result,
            is_error=is_error,
            context=current_context,
        )
        after_result = config.after_tool_call(after_ctx, signal)
        if after_result:
            if after_result.content is not None:
                content = after_result.content
            if after_result.is_error is not None:
                is_error = after_result.is_error

    result_msg = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content="".join(c.text for c in content if isinstance(c, TextContent))
        if content
        else "",
        is_error=is_error,
    )
    emit(
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result_msg,
            is_error=is_error,
        )
    )
    return result_msg


def _is_cancelled(signal: CancellationSignal | None) -> bool:
    return bool(signal and signal.is_cancelled())


def _cancel_reason(signal: CancellationSignal | None) -> str:
    if signal is None:
        return "interrupted by user"
    return signal.reason
