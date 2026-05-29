from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

"""Agent 核心循环。

基于 TS pi/packages/agent/src/agent-loop.ts。纯函数式，不持有状态。
流程：
  外层循环：followUp 驱动（队列为空时停止）
  内层循环：steer + tool call 驱动
    → stream_assistant_response()
    → execute_tool_calls()（按 execution_mode 串行/并行）
    → prepareNextTurn / shouldStopAfterTurn
"""

from .types import (  # noqa: E402
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
)


# 事件辅助构造
def _agent_start_event() -> AgentEvent:
    return {"type": "agent_start"}  # type: ignore


def _agent_end_event(messages=None) -> AgentEvent:
    return {"type": "agent_end", "messages": messages or []}  # type: ignore


def _turn_start_event() -> AgentEvent:
    return {"type": "turn_start"}  # type: ignore


def _turn_end_event(message=None, tool_results=None) -> AgentEvent:
    return {"type": "turn_end", "message": message, "tool_results": tool_results or []}  # type: ignore


def _message_start_event(message) -> AgentEvent:
    return {"type": "message_start", "message": message}  # type: ignore


def _message_end_event(message) -> AgentEvent:
    return {"type": "message_end", "message": message}  # type: ignore


def _message_update_event(message) -> AgentEvent:
    return {"type": "message_update", "message": message}  # type: ignore


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: Any = None,
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
    signal: Any = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if getattr(last, "role", "") == "assistant":
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
    signal: Any = None,
    stream_fn: StreamFn | None = None,
) -> None:
    current_context = initial_context
    config = initial_config
    first_turn = True
    pending_messages: list[AgentMessage] = []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
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
    signal: Any,
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
    signal: Any,
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
        if signal and hasattr(signal, "aborted") and signal.aborted:
            break
    return ExecutedToolBatch(results=results, terminate=False)


async def _execute_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: Any,
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
    signal: Any,
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

    if config.before_tool_call:
        ctx = BeforeToolCallContext(
            assistant_message=assistant_message,
            tool_call=tool_call,
            args=args,
            context=current_context,
        )
        before_result = config.before_tool_call(ctx, signal)
        if before_result and getattr(before_result, "block", False):
            return ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=getattr(before_result, "reason", "")
                or "Tool execution was blocked",
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
        content="".join(getattr(c, "text", "") for c in content) if content else "",
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
    signal: Any,
    emit: Callable[[AgentEvent], None],
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    messages = context.messages

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
            et = ""
            delta_text = ""
            reasoning_delta = ""
            calls_list = []

            if isinstance(event, dict):
                et = event.get("type", "")
                if et == "text_delta":
                    delta_text = event.get("delta", "")
                elif et == "reasoning_delta":
                    reasoning_delta = event.get("delta", "")
                elif et == "tool_call_ready":
                    calls_list = event.get("calls", [])
            else:
                cls_name = event.__class__.__name__
                if cls_name == "TextDelta":
                    et = "text_delta"
                    delta_text = getattr(event, "chunk", "")
                elif cls_name == "ReasoningDelta":
                    et = "reasoning_delta"
                    reasoning_delta = getattr(event, "chunk", "")
                elif cls_name == "ToolCallReady":
                    et = "tool_call_ready"
                    calls_list = getattr(event, "calls", [])

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
                    call_id = (
                        call.get("id", "")
                        if isinstance(call, dict)
                        else getattr(call, "id", "")
                    )
                    call_name = (
                        call.get("name", "")
                        if isinstance(call, dict)
                        else getattr(call, "name", "")
                    )
                    call_input = (
                        call.get("input", {})
                        if isinstance(call, dict)
                        else getattr(call, "input", {})
                    )
                    tool_calls_found.append(
                        ToolCallBlock(
                            id=call_id,
                            name=call_name,
                            arguments=call_input
                            if isinstance(call_input, dict)
                            else {},
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
