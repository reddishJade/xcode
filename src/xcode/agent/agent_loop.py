"""Agent 核心循环。

Xcode 的类型化 Agent 循环。模块本身不持有运行状态。
流程：
  外层循环：follow-up 队列驱动（队列为空时停止）
  内层循环：steer + tool call 驱动
    → stream_assistant_response()
    → execute_tool_calls()（按 execution_mode 串行/并行）
    → prepare_next_turn / should_stop_after_turn
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..harness.agent_runtime.cancellation import CancellationToken
from ..harness.agent_runtime.events import (
    TextDelta,
    ReasoningDelta,
    ToolCallReady,
)
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    BeforeToolCallContext,
    ContentBlock,
    ShouldStopAfterTurnContext,
    StreamFn,
    TextContent,
    ToolCallBlock,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessage,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
)


# 事件辅助构造
def _agent_start_event() -> AgentStartEvent:
    return AgentStartEvent()


def _agent_end_event(messages=None) -> AgentEndEvent:
    return AgentEndEvent(messages=messages or [])


def _turn_start_event() -> TurnStartEvent:
    return TurnStartEvent()


def _turn_end_event(message=None, tool_results=None) -> TurnEndEvent:
    return TurnEndEvent(message=message, tool_results=tool_results or [])


def _message_start_event(message) -> MessageStartEvent:
    return MessageStartEvent(message=message)


def _message_end_event(message) -> MessageEndEvent:
    return MessageEndEvent(message=message)


def _message_update_event(message) -> MessageUpdateEvent:
    return MessageUpdateEvent(message=message)


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=list(context.tools) if context.tools else [],
    )

    emit(_agent_start_event())
    emit(_turn_start_event())
    for prompt in prompts:
        emit(_message_start_event(prompt))
        emit(_message_end_event(prompt))

    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if isinstance(last, AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=list(context.tools) if context.tools else [],
    )

    emit(_agent_start_event())
    emit(_turn_start_event())

    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> None:
    current_context = initial_context
    config = initial_config
    first_turn = True
    pending_messages: list[AgentMessage] = []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if _is_cancelled(signal):
                message = _cancelled_message(signal)
                new_messages.append(message)
                emit(_message_start_event(message))
                emit(_message_end_event(message))
                emit(_turn_end_event(message, []))
                emit(_agent_end_event(new_messages))
                return

            if not first_turn:
                emit(_turn_start_event())
            else:
                first_turn = False

            if pending_messages:
                for msg in pending_messages:
                    emit(_message_start_event(msg))
                    emit(_message_end_event(msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            message = await _stream_assistant_response(
                current_context, config, signal, emit, stream_fn
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                emit(_turn_end_event(message, []))
                emit(_agent_end_event(new_messages))
                return

            tool_calls = [b for b in message.content if isinstance(b, ToolCallBlock)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                executed = await _execute_tool_calls(
                    current_context, message, tool_calls, config, signal, emit
                )
                tool_results.extend(executed.results)
                has_more_tool_calls = not executed.terminate

                for tr in executed.results:
                    current_context.messages.append(tr)
                    new_messages.append(tr)

            emit(_turn_end_event(message, tool_results))

            if config.prepare_next_turn:
                update = config.prepare_next_turn()
                if update and update.context:
                    current_context = update.context

            if config.should_stop_after_turn:
                ctx = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=new_messages,
                )
                if config.should_stop_after_turn(ctx):
                    emit(_agent_end_event(new_messages))
                    return

            if config.get_steering_messages:
                pending_messages = config.get_steering_messages() or []
            else:
                pending_messages = []

        if config.get_follow_up_messages:
            follow_up = config.get_follow_up_messages()
            if follow_up:
                pending_messages = follow_up
                continue

        break

    emit(_agent_end_event(new_messages))


@dataclass
class ExecutedToolBatch:
    results: list[ToolResultMessage]
    terminate: bool


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    has_sequential = False
    for tc in tool_calls:
        if isinstance(tc, ToolCallBlock):
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
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    results: list[ToolResultMessage] = []
    for tc in tool_calls:
        if not isinstance(tc, ToolCallBlock):
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
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedToolBatch:
    tasks = []
    for tc in tool_calls:
        if not isinstance(tc, ToolCallBlock):
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
    results: list[ToolResultMessage] = []
    for r in raw:
        if isinstance(r, ToolResultMessage):
            results.append(r)
    return ExecutedToolBatch(results=results, terminate=False)


async def _execute_one(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallBlock,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
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


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: Callable[[AgentEvent], None],
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    messages = context.messages
    if _is_cancelled(signal):
        return _cancelled_message(signal)

    if config.transform_context:
        messages = config.transform_context(messages, signal)

    convert_fn = config.convert_to_llm or (lambda msgs: [])
    llm_messages = convert_fn(messages)

    llm_context = {
        "system_prompt": context.system_prompt,
        "messages": llm_messages,
        "tools": context.tools,
    }

    if config.get_api_key:
        k = config.get_api_key(getattr(config.model, "provider", ""))
        if k:
            pass

    provider = getattr(config.model, "provider_obj", None)
    if provider:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_found: list[ToolCallBlock] = []

        dummy = AssistantMessage(content=[])
        emit(_message_start_event(dummy))

        tool_specs = _tools_to_specs(context.tools)
        async for event in provider.stream(tool_specs, llm_context):
            if _is_cancelled(signal):
                final = _cancelled_message(signal)
                emit(_message_end_event(final))
                return final

            et = ""
            delta_text = ""
            reasoning_delta = ""
            calls_list = []

            if isinstance(event, TextDelta):
                et = "text_delta"
                delta_text = event.chunk
            elif isinstance(event, ReasoningDelta):
                et = "reasoning_delta"
                reasoning_delta = event.chunk
            elif isinstance(event, ToolCallReady):
                et = "tool_call_ready"
                calls_list = event.calls

            if et == "text_delta" and delta_text:
                text_parts.append(delta_text)
                emit(
                    _message_update_event(
                        AssistantMessage(
                            content=[TextContent(text="".join(text_parts))],
                        )
                    )
                )
            elif et == "reasoning_delta" and reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            elif et == "tool_call_ready" and calls_list:
                for call in calls_list:
                    tool_calls_found.append(
                        ToolCallBlock(
                            id=call.id,
                            name=call.name,
                            arguments=dict(call.input),
                        )
                    )

        final_text = "".join(text_parts)
        content_blocks: list[ContentBlock] = [TextContent(text=final_text)]
        content_blocks.extend(tool_calls_found)
        final = AssistantMessage(
            content=content_blocks,
            reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            stop_reason="end_turn",
        )
        if not tool_calls_found:
            context.messages.append(final)
        emit(_message_end_event(final))
        return final
    else:
        msg = AssistantMessage(
            content=[TextContent(text="No provider configured")],
            stop_reason="end_turn",
        )
        emit(_message_start_event(msg))
        emit(_message_end_event(msg))
        return msg


def _tools_to_specs(tools: list[AgentTool[Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {"name": t.name, "description": t.description, "schema": t.parameters}
        for t in tools
    ]


def _is_cancelled(signal: CancellationToken | None) -> bool:
    return bool(signal and signal.is_cancelled())


def _cancel_reason(signal: CancellationToken | None) -> str:
    if signal is None:
        return "interrupted by user"
    return signal.reason


def _cancelled_message(signal: CancellationToken | None) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        stop_reason="aborted",
        error_message=_cancel_reason(signal),
    )
