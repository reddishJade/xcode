"""Agent 工具执行。

从 agent_loop.py 提取的工具调用执行逻辑：串行/并行调度、
单工具执行、before/after 钩子。

**提取的设计原因**：
- 关注点分离：agent_loop.py 专注于轮次管理，tool_execution.py 专注于工具调度
- 测试隔离：工具执行逻辑可以独立测试，不依赖完整 agent 循环
- 复用性：其他执行模式（如 plan mode）可以复用工具执行逻辑
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xcode.agent.types import ImageContent, TextContent, ToolCallContent
from .config import AfterToolCallContext, AgentContext, AgentLoopConfig, BeforeToolCallContext
from .events import AgentEvent, ToolExecutionEndEvent, ToolExecutionStartEvent, ToolExecutionUpdateEvent
from .messages import AssistantMessage, ToolResultMessage
from .protocols import AgentToolResult, CancellationSignal

logger = logging.getLogger(__name__)


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
    """根据 execution_mode 调度串行或并行工具执行。

    串行/并行调度设计原因：
    - 默认并行：充分利用并发提升效率（如多文件读取、并行搜索）
    - 强制串行：某些工具有副作用依赖（如先写文件再读取）或需要顺序保证
    - 批次隔离：同一批次内的工具可以并行，不同批次按顺序执行

    分批策略（partition_tool_calls_for_execution）：
    按工具的 execution_mode 元数据分组，sequential 工具单独成批，
    其余工具尽可能合并到同一批次并行执行。
    """
    if config.tool_execution == "sequential":
        return await _execute_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )

    results: list[ToolResultMessage] = []
    terminate_flags: list[bool] = []
    for batch in partition_tool_calls_for_execution(current_context, tool_calls):
        if (
            len(batch) == 1
            and _tool_execution_mode(current_context, batch[0]) == "sequential"
        ):
            executed = await _execute_sequential(
                current_context, assistant_message, batch, config, signal, emit
            )
        else:
            executed = await _execute_parallel(
                current_context, assistant_message, batch, config, signal, emit
            )
        results.extend(executed.results)
        terminate_flags.append(executed.terminate)
        if _is_cancelled(signal):
            break

    all_terminate = len(terminate_flags) > 0 and all(terminate_flags)
    return ExecutedToolBatch(results=results, terminate=all_terminate)


async def _execute_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    results: list[ToolResultMessage] = []
    terminate_flags: list[bool] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, ToolCallContent):
            continue
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        result, terminate = await _execute_one(
            current_context, assistant_message, tool_call, config, signal, emit
        )
        results.append(result)
        terminate_flags.append(terminate)
        if _is_cancelled(signal):
            break
    all_terminate = len(terminate_flags) > 0 and all(terminate_flags)
    return ExecutedToolBatch(results=results, terminate=all_terminate)


async def _execute_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    tasks = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, ToolCallContent):
            continue
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        tasks.append(
            _execute_one(
                current_context, assistant_message, tool_call, config, signal, emit
            )
        )

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[ToolResultMessage] = []
    terminate_flags: list[bool] = []
    for raw_result in raw_results:
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            results.append(raw_result[0])
            terminate_flags.append(raw_result[1])
        elif isinstance(raw_result, BaseException):
            logger.exception(
                "Tool execution raised unexpected exception", exc_info=raw_result
            )
    all_terminate = len(terminate_flags) > 0 and all(terminate_flags)
    return ExecutedToolBatch(results=results, terminate=all_terminate)


async def _execute_one(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> tuple[ToolResultMessage, bool]:
    """Execute a single tool call. Returns (result_message, terminate)."""
    try:
        return await _execute_one_impl(
            current_context, assistant_message, tool_call, config, signal, emit
        )
    except BaseException:
        logger.exception("Unexpected error executing tool %s", tool_call.name)
        return _error_result(tool_call, "unexpected tool execution error")


async def _execute_one_impl(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> tuple[ToolResultMessage, bool]:
    """Execute a single tool call. Returns (result_message, terminate)."""
    tool = _find_tool(current_context, tool_call)

    if tool is None:
        return _error_result(tool_call, f"unknown tool: {tool_call.name}")

    args = tool_call.arguments or {}

    if _is_cancelled(signal):
        return _error_result(tool_call, _cancel_reason(signal))

    before_block = _run_before_tool_hook(
        current_context, assistant_message, tool_call, args, config, signal
    )
    if before_block is not None:
        return before_block

    def _on_update(partial: AgentToolResult) -> None:
        emit(
            ToolExecutionUpdateEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=args,
                partial_result=partial,
            )
        )

    tool_result, content, is_error, terminate = await _run_tool_handler(
        tool, tool_call, args, signal, _on_update
    )

    content, is_error, terminate = _run_after_tool_hook(
        current_context,
        assistant_message,
        tool_call,
        args,
        tool_result,
        is_error,
        terminate,
        content,
        config,
        signal,
    )

    result_msg = _tool_result_message(tool_call, content, is_error)
    _emit_tool_end(tool_call, result_msg, is_error, emit)
    return result_msg, terminate


def _find_tool(current_context: AgentContext, tool_call: ToolCallContent) -> Any:
    for candidate_tool in current_context.tools or []:
        if candidate_tool.name == tool_call.name:
            return candidate_tool
    return None


def _error_result(
    tool_call: ToolCallContent,
    content: str,
) -> tuple[ToolResultMessage, bool]:
    return (
        ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=content,
            is_error=True,
        ),
        False,
    )


def _run_before_tool_hook(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    args: dict[str, Any],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
) -> tuple[ToolResultMessage, bool] | None:
    if not config.before_tool_call:
        return None
    ctx = BeforeToolCallContext(
        assistant_message=assistant_message,
        tool_call=tool_call,
        args=args,
        context=current_context,
    )
    before_result = config.before_tool_call(ctx, signal)
    if before_result and before_result.block:
        return _error_result(
            tool_call,
            before_result.reason or "Tool execution was blocked",
        )
    return None


async def _run_tool_handler(
    tool: Any,
    tool_call: ToolCallContent,
    args: dict[str, Any],
    signal: CancellationSignal | None,
    on_update: Callable[[AgentToolResult], None],
) -> tuple[AgentToolResult, list[TextContent | ImageContent], bool, bool]:
    try:
        tool_result = await tool.execute(
            tool_call.id, args, signal, on_update=on_update
        )
        return tool_result, tool_result.content, False, tool_result.terminate
    except Exception as e:
        tool_result = AgentToolResult(content=[TextContent(text=f"Tool error: {e}")])
        return tool_result, tool_result.content, True, False


def _run_after_tool_hook(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    args: dict[str, Any],
    tool_result: AgentToolResult,
    is_error: bool,
    terminate: bool,
    content: list[TextContent | ImageContent],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
) -> tuple[list[TextContent | ImageContent], bool, bool]:
    if not config.after_tool_call:
        return content, is_error, terminate
    after_ctx = AfterToolCallContext(
        assistant_message=assistant_message,
        tool_call=tool_call,
        args=args,
        result=tool_result,
        is_error=is_error,
        context=current_context,
    )
    after_result = config.after_tool_call(after_ctx, signal)
    if after_result is None:
        return content, is_error, terminate
    if after_result.content is not None:
        content = after_result.content
    if after_result.is_error is not None:
        is_error = after_result.is_error
    if after_result.terminate is not None:
        terminate = after_result.terminate
    return content, is_error, terminate


def _tool_result_message(
    tool_call: ToolCallContent,
    content: list[TextContent | ImageContent],
    is_error: bool,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content="".join(c.text for c in content if isinstance(c, TextContent))
        if content
        else "",
        is_error=is_error,
    )


def _emit_tool_end(
    tool_call: ToolCallContent,
    result_msg: ToolResultMessage,
    is_error: bool,
    emit: Callable[[AgentEvent], None],
) -> None:
    emit(
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result_msg,
            is_error=is_error,
        )
    )


def _is_cancelled(signal: CancellationSignal | None) -> bool:
    return bool(signal and signal.is_cancelled())


def _cancel_reason(signal: CancellationSignal | None) -> str:
    if signal is None:
        return "interrupted by user"
    return signal.reason


def partition_tool_calls_for_execution(
    current_context: AgentContext,
    tool_calls: list[ToolCallContent],
) -> list[list[ToolCallContent]]:
    """按工具执行模式将连续并发调用分批。"""
    batches: list[list[ToolCallContent]] = []
    parallel_batch: list[ToolCallContent] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, ToolCallContent):
            continue
        if _tool_execution_mode(current_context, tool_call) == "parallel":
            parallel_batch.append(tool_call)
            continue
        if parallel_batch:
            batches.append(parallel_batch)
            parallel_batch = []
        batches.append([tool_call])
    if parallel_batch:
        batches.append(parallel_batch)
    return batches


def _tool_execution_mode(
    current_context: AgentContext,
    tool_call: ToolCallContent,
) -> str:
    """返回工具的执行模式（parallel 或 sequential）。

    默认 sequential 的设计原因：
    - 保守策略：避免副作用工具并发执行导致状态不一致
    - 例如：先 write_file 再 read_file，若并行则 read 可能读到旧内容
    - 工具可通过 metadata 显式声明 execution_mode="parallel" 允许并发
    """
    for tool in current_context.tools or []:
        if tool.name == tool_call.name:
            return tool.execution_mode or "sequential"
    return "sequential"
