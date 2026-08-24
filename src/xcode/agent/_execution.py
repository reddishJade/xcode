"""工具执行调度、参数校验、看门狗检测。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import jsonschema

from xcode.agent.config import (
    AfterToolCallContext,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    _LoopRunState,
)
from xcode.agent.events import (
    AgentEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from xcode.agent.messages import (
    AssistantMessage,
    ToolResultMessage,
    ToolResultMessageContent,
)
from xcode.agent.types import (
    AgentTool,
    AgentToolResult,
    CancellationSignal,
    TextContent,
    ToolArguments,
    ToolCallContent,
    ToolRenderIntent,
    ToolResultContentBlock,
    materialize_json_mapping,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutedToolBatch:
    results: list[ToolResultMessage]
    terminate: bool


# ── 取消检查（共享工具） ──

# 打断后等待可取消工具（如 bash 杀进程）自行收尾的宽限期。
_TOOL_CANCEL_GRACE_SECONDS = 5.0


def is_cancelled(signal: CancellationSignal | None) -> bool:
    return bool(signal and signal.is_cancelled())


def cancel_reason(signal: CancellationSignal | None) -> str:
    if signal is None:
        return "interrupted by user"
    return signal.reason


async def _wait_cancellation(signal: CancellationSignal) -> None:
    """轮询取消信号，供工具执行竞争使用。"""
    while not is_cancelled(signal):
        await asyncio.sleep(0.01)


# ── 工具执行调度 ──


def partition_tool_calls_for_execution(
    current_context: AgentContext,
    tool_calls: list[ToolCallContent],
) -> list[list[ToolCallContent]]:
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
    for tool in current_context.tools or []:
        if tool.name == tool_call.name:
            return tool.execution_mode or "sequential"
    return "sequential"


def validate_tool_arguments(
    tool: AgentTool,
    tool_call: ToolCallContent,
    args: ToolArguments,
) -> str | None:
    schema = materialize_json_mapping(tool.parameters)
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.SchemaError as exc:
        return f"tool schema error for {tool_call.name}: {exc.message}"
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
        async with semaphore:
            return await _execute_one(
                current_context,
                assistant_message,
                tool_call,
                config,
                signal,
                emit,
            )

    pending: dict[asyncio.Task[tuple[ToolResultMessage, bool]], ToolCallContent] = {}
    for tool_call in tool_calls:
        emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments or {},
            )
        )
        pending[asyncio.create_task(execute_limited(tool_call))] = tool_call

    completed: dict[str, tuple[ToolResultMessage, bool]] = {}
    while pending:
        if is_cancelled(signal):
            # 打断：先给可取消工具（如 bash 杀进程）宽限期自行收尾，
            # 宽限期后仍未完成的工具直接放弃并返回打断结果。
            done, still_pending = await asyncio.wait(
                pending, timeout=_TOOL_CANCEL_GRACE_SECONDS
            )
            for task in done:
                _append_execution_result(task, pending[task], completed)
            for task in still_pending:
                tool_call = pending[task]
                task.cancel()
                interrupted, interrupted_terminate = _error_result(
                    tool_call, cancel_reason(signal)
                )
                completed[tool_call.id] = (interrupted, interrupted_terminate)
                _emit_tool_end(tool_call, interrupted, True, emit)
            break
        done, pending_set = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            _append_execution_result(task, pending[task], completed)
        pending = {task: pending[task] for task in pending_set}

    ordered = [completed[call.id] for call in tool_calls if call.id in completed]
    results = [entry[0] for entry in ordered]
    terminate_flags = [entry[1] for entry in ordered]
    all_terminate = len(terminate_flags) > 0 and all(terminate_flags)
    return ExecutedToolBatch(results=results, terminate=all_terminate)


def _append_execution_result(
    task: asyncio.Task[tuple[ToolResultMessage, bool]],
    tool_call: ToolCallContent,
    completed: dict[str, tuple[ToolResultMessage, bool]],
) -> None:
    """把已完成任务的执行结果收集进批次；异常任务记录日志并跳过。"""
    try:
        result, terminate = task.result()
    except Exception:
        logger.exception("Tool execution raised unexpected exception")
        return
    completed[tool_call.id] = (result, terminate)


async def _execute_one(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    config: AgentLoopConfig,
    signal: CancellationSignal | None,
    emit: Callable[[AgentEvent], None],
) -> tuple[ToolResultMessage, bool]:
    result_msg: ToolResultMessage | None = None
    try:
        result_msg, terminate = await _execute_one_impl(
            current_context, assistant_message, tool_call, config, signal, emit
        )
        return result_msg, terminate
    except Exception:
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
        message = before_result.reason or "Tool execution was blocked"
        if before_result.suggestion:
            message = f"{message}\n\nSuggestion: {before_result.suggestion}"
        return _error_result(tool_call, message)
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

    result_msg = _tool_result_message(
        tool_call,
        content,
        is_error,
        tool_result.details,
        tool_result.render_intent,
    )
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
    execution = tool.execute(tool_call.id, args, signal, on_update=on_update)
    tool_task = asyncio.create_task(
        _await_tool_execution(execution, tool_call, timeout_seconds)
    )
    if signal is None:
        return await tool_task

    cancel_waiter = asyncio.create_task(_wait_cancellation(signal))
    try:
        done, _pending = await asyncio.wait(
            {tool_task, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        if tool_task in done:
            return tool_task.result()
        # 打断：先给可感知取消的工具（如 bash 杀进程）宽限期自行收尾，
        # 保留其真实输出；宽限期后仍未返回则放弃等待并给出打断结果。
        try:
            return await asyncio.wait_for(
                asyncio.shield(tool_task), timeout=_TOOL_CANCEL_GRACE_SECONDS
            )
        except TimeoutError:
            tool_task.cancel()
            interrupted = AgentToolResult(
                content=[TextContent(text=cancel_reason(signal))],
                is_error=True,
            )
            return interrupted, interrupted.content, True, False
    finally:
        cancel_waiter.cancel()


async def _await_tool_execution(
    execution: Awaitable[AgentToolResult],
    tool_call: ToolCallContent,
    timeout_seconds: float | None,
) -> tuple[AgentToolResult, list[ToolResultContentBlock], bool, bool]:
    """等待工具执行并按超时/异常约定转换为结果。"""
    try:
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
                        f"Tool timed out after {timeout_seconds:g}s: {tool_call.name}"
                    )
                )
            ]
        )
        return tool_result, tool_result.content, True, False
    except (LookupError, OSError, RuntimeError, TypeError, ValueError) as e:
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
    render_intent: ToolRenderIntent | None = None,
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
        render_intent=render_intent,
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


# ── 工具看门狗 ──


DEFAULT_MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "edit_file",
        "bash",
        "create_file",
        "delete_file",
        "move_file",
        "rename_file",
    }
)

DEFAULT_READ_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "grep_search",
        "glob_files",
        "list_dir",
        "find_files",
    }
)


def tool_call_signature(call: ToolCallContent) -> str:
    args_str = json.dumps(call.arguments or {}, sort_keys=True, default=str)
    return f"{call.name}:{args_str}"


def tool_calls_signature(calls: list[ToolCallContent]) -> str:
    parts = [tool_call_signature(c) for c in calls]
    return "|".join(sorted(parts))


def is_file_mutation_tool(
    tool_name: str,
    mutation_tools: frozenset[str] | None = None,
) -> bool:
    return tool_name in (
        mutation_tools if mutation_tools is not None else DEFAULT_MUTATION_TOOLS
    )


def is_file_read_tool(
    tool_name: str,
    read_tools: frozenset[str] | None = None,
) -> bool:
    return tool_name in (read_tools if read_tools is not None else DEFAULT_READ_TOOLS)


def should_clear_read_history(
    new_calls: list[ToolCallContent],
    read_history: list[str],
    mutation_tools: frozenset[str] | None = None,
) -> bool:
    return any(is_file_mutation_tool(c.name, mutation_tools) for c in new_calls)


def is_tool_productive_default(
    tool_calls: list[ToolCallContent],
    tool_results: list[ToolResultMessage],
) -> bool:
    return any(not r.is_error for r in tool_results)


def update_repeated_tool_watchdog(
    state: _LoopRunState,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    tool_results: list[ToolResultMessage],
) -> str | None:
    counted_calls = [
        call
        for call in tool_calls
        if call.name not in config.watchdog_repeated_tool_skip
    ]
    if not counted_calls:
        state.repeated_tool_count = 0
        state.last_tool_signature = None
        return None

    counted_call_ids = {call.id for call in counted_calls}
    counted_results = [
        result for result in tool_results if result.tool_call_id in counted_call_ids
    ]

    if all(r.is_error for r in counted_results):
        state.repeated_tool_count = 0
        state.last_tool_signature = None
        return None

    sig = tool_calls_signature(counted_calls)
    if sig == state.last_tool_signature:
        state.repeated_tool_count += 1
    else:
        state.repeated_tool_count = 0
        state.last_tool_signature = sig

    if (
        config.watchdog_repeated_tool_limit > 0
        and state.repeated_tool_count >= config.watchdog_repeated_tool_limit
    ):
        return f"watchdog stopped repeated tool call: {counted_calls[0].name}"
    return None


def update_idle_tool_watchdog(
    state: _LoopRunState,
    tool_calls: list[ToolCallContent],
    tool_results: list[ToolResultMessage],
    config: AgentLoopConfig,
) -> str | None:
    is_productive = config.is_tool_productive or is_tool_productive_default
    if is_productive(tool_calls, tool_results):
        state.consecutive_idle_steps = 0
    else:
        state.consecutive_idle_steps += 1

    if (
        config.max_consecutive_idle_steps > 0
        and state.consecutive_idle_steps >= config.max_consecutive_idle_steps
    ):
        return (
            f"Watchdog triggered: {state.consecutive_idle_steps} consecutive steps "
            f"without productive tool calls."
        )
    return None
