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

import jsonschema

from xcode.agent.types import (
    TextContent,
    ToolArguments,
    ToolCallContent,
)
from .config import (
    AfterToolCallContext,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from .events import (
    AgentEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from .messages import AssistantMessage, ToolResultMessage, ToolResultMessageContent
from .protocols import (
    AgentTool,
    AgentToolResult,
    CancellationSignal,
    ToolResultContentBlock,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutedToolBatch:
    results: list[ToolResultMessage]
    terminate: bool


def partition_tool_calls_for_execution(
    current_context: AgentContext,
    tool_calls: list[ToolCallContent],
) -> list[list[ToolCallContent]]:
    """按工具执行模式将连续并发调用分批。"""
    batches: list[list[ToolCallContent]] = []
    parallel_batch: list[ToolCallContent] = []
    for tool_call in tool_calls:
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
    """返回工具声明的执行模式，未找到时保守地串行执行。"""
    for tool in current_context.tools or []:
        if tool.name == tool_call.name:
            return tool.execution_mode or "sequential"
    return "sequential"


def validate_tool_arguments(
    tool: AgentTool,
    tool_call: ToolCallContent,
    args: ToolArguments,
) -> str | None:
    """按工具 JSON schema 校验模型生成的参数。"""
    try:
        schema = dict(tool.parameters)
    except Exception as exc:
        return f"tool schema error for {tool_call.name}: {exc}"
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        path = (
            ".".join(str(part) for part in exc.absolute_path)
            if exc.absolute_path
            else tool_call.name
        )
        return f"tool argument schema error: {path}: {exc.message}"
    return None


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
        if is_cancelled(signal):
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
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments or {},
            )
        )
        result, terminate = await _execute_one(
            current_context, assistant_message, tool_call, config, signal, emit
        )
        results.append(result)
        terminate_flags.append(terminate)
        if is_cancelled(signal):
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
    semaphore = asyncio.Semaphore(max(1, config.tool_workers))

    async def execute_limited(
        tool_call: ToolCallContent,
    ) -> tuple[ToolResultMessage, bool]:
        """在共享并发额度内执行单个 parallel 工具。"""
        async with semaphore:
            return await _execute_one(
                current_context,
                assistant_message,
                tool_call,
                config,
                signal,
                emit,
            )

    tasks = []
    for tool_call in tool_calls:
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments or {},
            )
        )
        tasks.append(execute_limited(tool_call))

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
    """执行单个工具调用，返回结果消息和终止标记。"""
    result_msg: ToolResultMessage | None = None
    try:
        result_msg, terminate = await _execute_one_impl(
            current_context, assistant_message, tool_call, config, signal, emit
        )
        return result_msg, terminate
    except BaseException:
        logger.exception("Unexpected error executing tool %s", tool_call.name)
        result_msg, terminate = _error_result(
            tool_call, "unexpected tool execution error"
        )
        return result_msg, terminate
    finally:
        if result_msg is not None:
            _emit_tool_end(tool_call, result_msg, result_msg.is_error, emit)


async def _execute_one_impl(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> tuple[ToolResultMessage, bool]:
    """执行单个工具调用，返回结果消息和终止标记。"""
    tool = _find_tool(current_context, tool_call)

    if tool is None:
        return _error_result(tool_call, f"unknown tool: {tool_call.name}")

    args: ToolArguments = tool_call.arguments or {}
    if is_cancelled(signal):
        return _error_result(tool_call, cancel_reason(signal))

    before_result = _run_before_tool_hook(
        current_context, assistant_message, tool_call, args, config, signal
    )
    if before_result is not None and before_result.block:
        return _error_result(
            tool_call,
            before_result.reason or "Tool execution was blocked",
        )
    if before_result is not None and before_result.args is not None:
        args = before_result.args
        tool_call = tool_call.model_copy(update={"arguments": args})
        assistant_message.content = [
            (
                tool_call
                if isinstance(block, ToolCallContent) and block.id == tool_call.id
                else block
            )
            for block in assistant_message.content
        ]

    validation_error = validate_tool_arguments(tool, tool_call, args)
    if validation_error is not None:
        return _error_result(tool_call, validation_error)

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
        tool, tool_call, args, signal, _on_update, config.tool_timeout_seconds
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

    result_msg = _tool_result_message(tool_call, content, is_error, tool_result.details)
    return result_msg, terminate


def _find_tool(
    current_context: AgentContext, tool_call: ToolCallContent
) -> AgentTool | None:
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
    args: ToolArguments,
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
) -> BeforeToolCallResult | None:
    if not config.before_tool_call:
        return None
    ctx = BeforeToolCallContext(
        assistant_message=assistant_message,
        tool_call=tool_call,
        args=args,
        context=current_context,
    )
    return config.before_tool_call(ctx, signal)


async def _run_tool_handler(
    tool: AgentTool,
    tool_call: ToolCallContent,
    args: ToolArguments,
    signal: CancellationSignal | None,
    on_update: Callable[[AgentToolResult], None],
    timeout_seconds: float | None,
) -> tuple[AgentToolResult, list[ToolResultContentBlock], bool, bool]:
    try:
        execution = tool.execute(tool_call.id, args, signal, on_update=on_update)
        if timeout_seconds is not None and timeout_seconds > 0:
            tool_result = await asyncio.wait_for(execution, timeout=timeout_seconds)
        else:
            tool_result = await execution
        return (
            tool_result,
            tool_result.content,
            tool_result.is_error,
            tool_result.terminate,
        )
    except TimeoutError:
        tool_result = AgentToolResult(
            content=[
                TextContent(
                    text=(
                        f"Tool timed out after {timeout_seconds:g}s: "
                        f"{tool_call.name}"
                    )
                )
            ]
        )
        return tool_result, tool_result.content, True, False
    except Exception as e:
        tool_result = AgentToolResult(content=[TextContent(text=f"Tool error: {e}")])
        return tool_result, tool_result.content, True, False


def _run_after_tool_hook(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    args: ToolArguments,
    tool_result: AgentToolResult,
    is_error: bool,
    terminate: bool,
    content: list[ToolResultContentBlock],
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
) -> tuple[list[ToolResultContentBlock], bool, bool]:
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
    content: list[ToolResultContentBlock],
    is_error: bool,
    metadata: object = None,
) -> ToolResultMessage:
    result_content: ToolResultMessageContent
    if not content:
        result_content = ""
    elif any(not isinstance(item, TextContent) for item in content):
        result_content = content
    else:
        result_content = "".join(
            item.text for item in content if isinstance(item, TextContent)
        )
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result_content,
        is_error=is_error,
        metadata=metadata if isinstance(metadata, dict) else None,
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


def is_cancelled(signal: CancellationSignal | None) -> bool:
    return bool(signal and signal.is_cancelled())


def cancel_reason(signal: CancellationSignal | None) -> str:
    if signal is None:
        return "interrupted by user"
    return signal.reason
