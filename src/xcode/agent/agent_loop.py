"""Agent 核心循环。

Xcode 的类型化 Agent 循环。模块本身不持有运行状态。
流程：
  外层循环：步骤限制 + follow-up 队列驱动
  内层循环：compact → 模型调用 → 错误重试 → max_tokens 续写
    → stream_assistant_response()
    → execute_tool_calls()（按 execution_mode 串行/并行）
    → watchdog 检查 → prepare_next_turn / should_stop_after_turn
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    ReasoningDelta,
    StopReason,
    TextDelta,
    ToolCallEvent,
    UsageUpdate,
)
from xcode.ai.types import ToolDefinition
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopMetrics,
    AgentLoopResult,
    AgentMessage,
    AgentTool,
    AssistantMessage,
    CancellationSignal,
    CompactionEvent,
    ContentBlock,
    ShouldStopAfterTurnContext,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ThinkingUpdateEvent,
)
from .tool_execution import (
    ExecutedToolBatch,
    execute_tool_calls,
    _is_cancelled,
    _cancel_reason,
)


# ── 事件辅助构造 ──


def _agent_start_event() -> AgentStartEvent:
    return AgentStartEvent()


def _agent_end_event(
    messages: list[AgentMessage] | None = None,
    result: AgentLoopResult | None = None,
) -> AgentEndEvent:
    return AgentEndEvent(messages=messages or [])


def _turn_start_event() -> TurnStartEvent:
    return TurnStartEvent()


def _turn_end_event(
    message: AgentMessage | None = None,
    tool_results: list[ToolResultMessage] | None = None,
) -> TurnEndEvent:
    return TurnEndEvent(message=message, tool_results=tool_results or [])


def _message_start_event(message: AgentMessage) -> MessageStartEvent:
    return MessageStartEvent(message=message)


def _message_end_event(message: AgentMessage) -> MessageEndEvent:
    return MessageEndEvent(message=message)


def _message_update_event(message: AgentMessage) -> MessageUpdateEvent:
    return MessageUpdateEvent(message=message)


# ── 工具签名（用于重复工具看门狗）──


def _tool_signature(calls: list[ToolCallContent]) -> str:
    """生成工具调用签名，用于检测重复调用。"""
    parts = []
    for c in calls:
        args_str = json.dumps(c.arguments or {}, sort_keys=True, default=str)
        parts.append(f"{c.name}:{args_str}")
    return "|".join(sorted(parts))


def _is_tool_productive_default(
    tool_calls: list[ToolCallContent],
    tool_results: list[ToolResultMessage],
) -> bool:
    """默认生产力检查：有任何非错误结果即视为有生产力。"""
    return any(not r.is_error for r in tool_results)


# ── 公共 API ──


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None = None,
) -> AgentLoopResult:
    """运行 agent 核心循环。

    将 prompt 消息合并到 context 中，执行完整的 agent 循环，
    返回 AgentLoopResult 包含所有消息、步数和指标。
    """
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
    return await _run_loop(current_context, new_messages, config, emit, signal)


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None = None,
) -> AgentLoopResult:
    """从已有上下文继续运行 agent 循环。"""
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
    return await _run_loop(current_context, new_messages, config, emit, signal)


# ── 内部循环 ──


@dataclass
class _ProviderResponse:
    message: AssistantMessage
    stop_reason: StopReason


@dataclass
class _LoopRunState:
    first_turn: bool = True
    pending_messages: list[AgentMessage] = field(default_factory=list)
    last_tool_signature: str | None = None
    repeated_tool_count: int = 0
    consecutive_idle_steps: int = 0
    consecutive_continuations: int = 0
    step_retries: int = 0
    active_provider: Any = None


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None = None,
) -> AgentLoopResult:
    """核心 agent 循环。

    外层：步骤限制 + follow-up 队列驱动
    内层：compact → 模型调用 → 重试 → max_tokens → 工具执行 → watchdog
    """
    metrics = AgentLoopMetrics()
    current_context = initial_context
    state = _LoopRunState(active_provider=config.provider)

    for step in range(1, config.max_steps + 1):
        metrics.steps = step

        # ── 取消检查 ──
        if _is_cancelled(signal):
            return _finish_loop(
                new_messages,
                step,
                metrics,
                state.active_provider,
                emit,
            )

        # ── 处理 steer 队列 ──
        _append_steering_messages(current_context, new_messages, config)

        # ── 发出 turn 事件 ──
        if not state.first_turn:
            emit(_turn_start_event())
        else:
            state.first_turn = False

        # ── 处理 pending messages ──
        _drain_pending_messages(current_context, new_messages, state, emit)

        # ── 压缩检查 ──
        if config.should_compact and config.compact:
            if config.should_compact(current_context.messages):
                before = len(current_context.messages)
                current_context.messages = config.compact(current_context.messages)
                after = len(current_context.messages)
                emit(CompactionEvent(
                    messages_removed=before - after,
                    messages_after=after,
                    summary_token_estimate=0,
                    trigger="token_limit",
                    archive=None,
                ))

        # ── 内层循环：模型调用 + 重试 + max_tokens ──
        ctx_len_before = len(current_context.messages)
        inner_result = await _run_inner_loop(
            current_context,
            config,
            emit,
            signal,
            metrics,
            step,
            state,
        )

        if inner_result is None:
            return _finish_loop(
                new_messages,
                step,
                metrics,
                state.active_provider,
                emit,
                stopped_by_error=True,
            )

        message, stop_reason, new_provider = inner_result
        state.active_provider = new_provider

        for msg in current_context.messages[ctx_len_before:-1]:
            new_messages.append(msg)

        new_messages.append(message)
        metrics.llm_calls += 1

        # ── 错误/中止 → 退出 ──
        if stop_reason in ("error", "aborted"):
            emit(_turn_end_event(message, []))
            emit(_agent_end_event(new_messages))
            return AgentLoopResult(
                messages=new_messages,
                steps=step,
                metrics=metrics,
                active_provider=state.active_provider,
            )

        # ── 提取工具调用 ──
        tool_calls = [b for b in message.content if isinstance(b, ToolCallContent)]

        if not tool_calls:
            # 模型没有请求工具 → 本轮结束
            emit(_turn_end_event(message, []))

            # 检查 follow-up 队列
            if _queue_follow_up(state, config):
                continue
            return _finish_loop(
                new_messages,
                step,
                metrics,
                state.active_provider,
                emit,
            )

        # ── 工具执行 ──
        executed: ExecutedToolBatch = await execute_tool_calls(
            current_context, message, tool_calls, config, signal, emit
        )
        tool_results = executed.results
        _append_tool_results(current_context, new_messages, metrics, tool_results)

        emit(_turn_end_event(message, tool_results))

        # ── 重复工具看门狗 ──
        repeated_watchdog_reason = _update_repeated_tool_watchdog(
            state, tool_calls, config
        )
        if repeated_watchdog_reason:
            return _finish_loop(
                new_messages,
                step,
                metrics,
                state.active_provider,
                emit,
                stopped_by_watchdog=True,
                watchdog_reason=repeated_watchdog_reason,
            )

        # ── 空闲步骤看门狗 ──
        idle_watchdog_reason = _update_idle_tool_watchdog(
            state, tool_calls, tool_results, config
        )
        if idle_watchdog_reason:
            return _finish_loop(
                new_messages,
                step,
                metrics,
                state.active_provider,
                emit,
                stopped_by_watchdog=True,
                watchdog_reason=idle_watchdog_reason,
            )

        # ── prepare_next_turn 钩子 ──
        if config.prepare_next_turn:
            update = config.prepare_next_turn()
            if update and update.context:
                current_context = update.context

        # ── should_stop_after_turn 钩子 ──
        if config.should_stop_after_turn:
            ctx = ShouldStopAfterTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )
            if config.should_stop_after_turn(ctx):
                return _finish_loop(
                    new_messages,
                    step,
                    metrics,
                    state.active_provider,
                    emit,
                )

        # ── 处理 follow-up 队列 ──
        if _queue_follow_up(state, config):
            continue

        # 工具已执行但没有 follow-up，继续内层循环（下一轮工具调用）
        if not executed.terminate:
            continue

    # 步骤耗尽
    result = AgentLoopResult(
        messages=new_messages,
        steps=config.max_steps,
        stopped_by_limit=True,
        metrics=metrics,
        active_provider=state.active_provider,
    )
    emit(_agent_end_event(new_messages, result))
    return result


def _finish_loop(
    new_messages: list[AgentMessage],
    step: int,
    metrics: AgentLoopMetrics,
    active_provider: Any,
    emit: Callable[[AgentEvent], None],
    *,
    stopped_by_watchdog: bool = False,
    watchdog_reason: str | None = None,
    stopped_by_limit: bool = False,
    stopped_by_error: bool = False,
) -> AgentLoopResult:
    result = AgentLoopResult(
        messages=new_messages,
        steps=step,
        stopped_by_watchdog=stopped_by_watchdog,
        watchdog_reason=watchdog_reason,
        stopped_by_limit=stopped_by_limit,
        stopped_by_error=stopped_by_error,
        metrics=metrics,
        active_provider=active_provider,
    )
    emit(_agent_end_event(new_messages, result))
    return result


def _append_steering_messages(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
) -> None:
    if not config.get_steering_messages:
        return
    steer_msgs = config.get_steering_messages()
    if not steer_msgs:
        return
    for msg in steer_msgs:
        current_context.messages.append(msg)
        new_messages.append(msg)


def _drain_pending_messages(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    state: _LoopRunState,
    emit: Callable[[AgentEvent], None],
) -> None:
    if not state.pending_messages:
        return
    for msg in state.pending_messages:
        emit(_message_start_event(msg))
        emit(_message_end_event(msg))
        current_context.messages.append(msg)
        new_messages.append(msg)
    state.pending_messages = []


def _queue_follow_up(state: _LoopRunState, config: AgentLoopConfig) -> bool:
    if not config.get_follow_up_messages:
        return False
    follow_up = config.get_follow_up_messages()
    if not follow_up:
        return False
    state.pending_messages = follow_up
    return True


def _append_tool_results(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    metrics: AgentLoopMetrics,
    tool_results: list[ToolResultMessage],
) -> None:
    for tr in tool_results:
        current_context.messages.append(tr)
        new_messages.append(tr)
        metrics.tool_calls += 1


def _update_repeated_tool_watchdog(
    state: _LoopRunState,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
) -> str | None:
    sig = _tool_signature(tool_calls)
    if sig == state.last_tool_signature:
        state.repeated_tool_count += 1
    else:
        state.repeated_tool_count = 0
        state.last_tool_signature = sig

    if (
        config.watchdog_repeated_tool_limit > 0
        and state.repeated_tool_count >= config.watchdog_repeated_tool_limit
    ):
        return f"watchdog stopped repeated tool call: {tool_calls[0].name}"
    return None


def _update_idle_tool_watchdog(
    state: _LoopRunState,
    tool_calls: list[ToolCallContent],
    tool_results: list[ToolResultMessage],
    config: AgentLoopConfig,
) -> str | None:
    is_productive = config.is_tool_productive or _is_tool_productive_default
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


# ── 内层循环 ──


async def _run_inner_loop(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None,
    metrics: AgentLoopMetrics,
    step: int,
    state: _LoopRunState,
) -> tuple[AssistantMessage, str, Any] | None:
    """内层循环：模型调用 → 错误重试 → max_tokens 续写。

    通过 state 对象共享 step_retries / consecutive_continuations 计数器，
    无需外循环再手动同步。
    返回 (message, stop_reason, provider)，或 None 表示应提前退出。
    """
    provider = state.active_provider
    while True:
        if _is_cancelled(signal):
            return _cancelled_message(signal), "aborted", provider

        if provider is None:
            msg = AssistantMessage(
                content=[TextContent(text="No provider configured")],
                stop_reason="end_turn",
            )
            emit(_message_start_event(msg))
            emit(_message_end_event(msg))
            return msg, "end_turn", provider

        response = await _call_provider(
            context,
            config,
            emit,
            signal,
            metrics,
            provider,
        )
        message = response.message
        stop_reason = response.stop_reason

        # ── 检查是否为 FinalMessage 的错误 ──
        if stop_reason == "error":
            state.step_retries += 1
            should_retry, fallback_message = await _handle_provider_error(
                message, state.step_retries, config, emit
            )
            if should_retry:
                continue
            if fallback_message is not None:
                return None
            emit(_message_end_event(message))
            return message, stop_reason, provider

        # ── max_tokens 续写 ──
        if _should_continue_max_tokens(stop_reason, config):
            state.consecutive_continuations = _update_continuation_count(
                message, state.consecutive_continuations, config
            )
            _append_continuation_prompt(context, message)
            continue

        # ── 正常结束 ──
        context.messages.append(message)
        emit(_message_end_event(message))
        state.step_retries = 0
        state.consecutive_continuations = 0
        return message, stop_reason, provider


async def _call_provider(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None,
    metrics: AgentLoopMetrics,
    provider: Any,
) -> _ProviderResponse:
    messages = context.messages
    if config.transform_context:
        messages = config.transform_context(messages, signal)

    convert_fn = config.convert_to_llm or (lambda msgs: [])
    llm_messages = convert_fn(messages)
    tool_definitions = _tools_to_definitions(context.tools)

    started = perf_counter()
    events = await _collect_provider_events(
        provider,
        llm_messages,
        tool_definitions,
        config,
    )
    elapsed = round((perf_counter() - started) * 1000, 3)
    metrics.model_latencies_ms.append(elapsed)
    return _provider_events_to_response(events, metrics, emit)


async def _collect_provider_events(
    provider: Any,
    llm_messages: list[Any],
    tool_definitions: list[ToolDefinition],
    config: AgentLoopConfig,
) -> list[ProviderEvent]:
    try:
        events: list[ProviderEvent] = []
        kwargs = {}
        if config.options is not None:
            kwargs["options"] = config.options
        async for event in provider.stream(llm_messages, tool_definitions, **kwargs):
            events.append(event)
        return events
    except Exception as e:
        return [FinalMessage(f"Provider error: {e}", "error")]


def _provider_events_to_response(
    events: list[ProviderEvent],
    metrics: AgentLoopMetrics,
    emit: Callable[[AgentEvent], None],
) -> _ProviderResponse:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_found: list[ToolCallContent] = []
    stop_reason: StopReason = "end_turn"

    for event in events:
        if isinstance(event, TextDelta):
            _append_text_delta(text_parts, event, emit)
        elif isinstance(event, ReasoningDelta):
            reasoning_parts.append(event.chunk)
            emit(ThinkingUpdateEvent(reasoning_content=event.chunk))
        elif isinstance(event, ToolCallEvent):
            tool_calls_found.extend(_tool_call_content_blocks(event))
        elif isinstance(event, UsageUpdate):
            metrics.input_tokens += event.input_tokens
            metrics.output_tokens += event.output_tokens
        if isinstance(event, FinalMessage):
            stop_reason = event.stop_reason or "end_turn"

    content_blocks: list[ContentBlock] = [TextContent(text="".join(text_parts))]
    content_blocks.extend(tool_calls_found)
    return _ProviderResponse(
        message=AssistantMessage(
            content=content_blocks,
            reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            stop_reason=stop_reason,
        ),
        stop_reason=stop_reason,
    )


def _append_text_delta(
    text_parts: list[str],
    event: TextDelta,
    emit: Callable[[AgentEvent], None],
) -> None:
    text_parts.append(event.chunk)
    emit(
        _message_update_event(
            AssistantMessage(
                content=[TextContent(text="".join(text_parts))],
            )
        )
    )


def _tool_call_content_blocks(event: ToolCallEvent) -> list[ToolCallContent]:
    return [
        ToolCallContent(
            id=call.id,
            name=call.name,
            arguments=dict(call.input),
        )
        for call in event.calls
    ]


async def _handle_provider_error(
    message: AssistantMessage,
    step_retries: int,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
) -> tuple[bool, AssistantMessage | None]:
    if step_retries <= config.max_step_retries:
        delay = config.retry_backoff_base * (2 ** (step_retries - 1))
        await asyncio.sleep(delay)
        return True, None
    if _has_empty_text_response(message):
        msg = AssistantMessage(
            content=[TextContent(text="I encountered an error.")],
            stop_reason="error",
        )
        emit(_message_start_event(msg))
        emit(_message_end_event(msg))
        return False, msg
    return False, None


def _has_empty_text_response(message: AssistantMessage) -> bool:
    content_blocks = message.content
    return not content_blocks or (
        len(content_blocks) == 1
        and isinstance(content_blocks[0], TextContent)
        and not content_blocks[0].text
    )


def _should_continue_max_tokens(
    stop_reason: str,
    config: AgentLoopConfig,
) -> bool:
    return stop_reason == "max_tokens" and config.max_tokens_continuation


def _update_continuation_count(
    message: AssistantMessage,
    consecutive_continuations: int,
    config: AgentLoopConfig,
) -> int:
    inc = _estimate_text_tokens(
        json.dumps(message.content, ensure_ascii=False, default=str)
    )
    updated = (
        consecutive_continuations + 1 if inc < config.min_continuation_tokens else 0
    )
    if updated >= config.max_consecutive_continuations:
        raise RuntimeError(
            "Diminishing Returns: consecutive output token increments "
            f"below {config.min_continuation_tokens} limit."
        )
    return updated


def _append_continuation_prompt(
    context: AgentContext,
    message: AssistantMessage,
) -> None:
    context.messages.append(message)
    context.messages.append(UserMessage(content="continue"))


# ── 辅助函数 ──


def _tools_to_definitions(tools: list[AgentTool[Any]] | None) -> list[ToolDefinition]:
    if not tools:
        return []
    return [
        ToolDefinition(name=t.name, description=t.description, schema=t.parameters)
        for t in tools
    ]


def _cancelled_message(signal: CancellationSignal | None) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        stop_reason="aborted",
        error_message=_cancel_reason(signal),
    )


def _estimate_text_tokens(text: str) -> int:
    """粗略估算文本 token 数（约 4 字符/token）。"""
    return max(1, len(text) // 4)
