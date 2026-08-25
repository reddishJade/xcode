"""Agent 核心循环。

Xcode 的类型化 Agent 循环。模块本身不持有运行状态。

**循环设计原因**：
- 外层循环（步骤限制）：防止单次 run 无限执行
- 内层循环（compact → 模型调用 → 错误重试 → max_tokens 续写）：应对上下文溢出和部分生成
- 可注入设计：所有外部依赖（provider、工具、hooks）通过参数注入，便于测试和替换

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
import time
from collections.abc import Callable

from xcode.agent.types import CancellationSignal, TextContent, ToolCallContent
from xcode.ai.providers.base import StreamProvider

from ._compaction import estimate_tokens
from ._execution import (
    ExecutedToolBatch,
    cancel_reason,
    execute_tool_calls,
    is_cancelled,
    update_idle_tool_watchdog,
    update_repeated_tool_watchdog,
)
from ._provider import call_provider
from .config import (
    AgentContext,
    AgentLoopConfig,
    ShouldStopAfterTurnContext,
    _LoopRunState,
)
from .events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    CompactionArchive,
    CompactionEvent,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from .results import AgentLoopMetrics, AgentLoopResult, TerminationReason

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


# ── 公共 API ──


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None = None,
    steer_queue: Callable[[], list[AgentMessage]] | None = None,
    finish_steering: Callable[[], list[AgentMessage]] | None = None,
    reopen_steering: Callable[[], None] | None = None,
) -> AgentLoopResult:
    """运行 agent 核心循环。

    将 prompt 消息合并到 context 中，执行完整的 agent 循环，
    返回 AgentLoopResult 包含所有消息、步数和指标。
    """
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        request_prefix=list(context.request_prefix),
        messages=list(context.messages) + list(prompts),
        tools=list(context.tools) if context.tools else [],
        context_state=context.context_state,
        context_manager=context.context_manager,
        state=dict(context.state),
        project_root=context.project_root,
        cwd=context.cwd,
        request_token_budget=(
            config.request_token_budget
            if config.request_token_budget > 0
            else context.request_token_budget
        ),
    )
    if current_context.context_manager is not None:
        current_context.context_manager.token_usage.context_budget = (
            current_context.request_token_budget
        )
    emit(_agent_start_event())
    emit(_turn_start_event())
    for prompt in prompts:
        emit(_message_start_event(prompt))
        emit(_message_end_event(prompt))
    result = await _run_loop(
        current_context,
        new_messages,
        config,
        emit,
        signal,
        steer_queue=steer_queue,
        finish_steering=finish_steering,
        reopen_steering=reopen_steering,
    )
    if current_context.context_manager is not None:
        current_context.context_manager.replace_history(result.surface)
    return result


# ── 外层循环 ──


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None = None,
    steer_queue: Callable[[], list[AgentMessage]] | None = None,
    finish_steering: Callable[[], list[AgentMessage]] | None = None,
    reopen_steering: Callable[[], None] | None = None,
) -> AgentLoopResult:
    """核心 agent 循环。

    外层：持续运行（可选步骤限制）
    内层：compact → 模型调用 → 重试 → max_tokens → 工具执行 → watchdog
    """
    metrics = AgentLoopMetrics()
    current_context = initial_context
    state = _LoopRunState(active_provider=config.provider)

    step = 0
    while config.max_steps is None or step < config.max_steps:
        step += 1
        metrics.steps = step

        # ── 取消检查 ──
        if is_cancelled(signal):
            return _finish_loop(
                new_messages,
                current_context.messages,
                step,
                metrics,
                state.active_provider,
                emit,
                termination_reason=TerminationReason.CANCELLED,
                error_detail=cancel_reason(signal),
            )

        # ── 处理 steer 队列 ──
        _append_steering_messages(current_context, new_messages, steer_queue)

        # ── 发出 turn 事件 ──
        if not state.first_turn:
            emit(_turn_start_event())
        else:
            state.first_turn = False

        # ── 压缩检查 ──
        if (
            config.should_compact
            and config.compact
            and config.should_compact(current_context.messages)
        ):
            messages_before = list(current_context.messages)
            before = len(messages_before)
            compacted_messages = config.compact(current_context.messages)
            if current_context.context_manager is not None:
                current_context.messages = (
                    current_context.context_manager.complete_compaction(
                        compacted_messages,
                        before_messages=before,
                    )
                )
            else:
                current_context.messages = compacted_messages
                current_context.context_state.reset()
            after = len(current_context.messages)
            archive: CompactionArchive | None = None
            if config.archive_writer:
                archive_path = config.archive_writer(messages_before)
                if archive_path:
                    archive = CompactionArchive(path=archive_path, status="summary")
            emit(
                CompactionEvent(
                    messages_removed=before - after,
                    messages_after=after,
                    summary_token_estimate=0,
                    trigger="token_limit",
                    archive=archive,
                    replacement=list(current_context.messages),
                )
            )

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
                current_context.messages,
                step,
                metrics,
                state.active_provider,
                emit,
                termination_reason=TerminationReason.PROVIDER_ERROR,
            )

        message, stop_reason, new_provider = inner_result
        state.active_provider = new_provider

        new_messages.extend(current_context.messages[ctx_len_before:-1])

        new_messages.append(message)
        metrics.llm_calls += 1

        # ── 错误/中止 → 退出 ──
        if stop_reason in ("error", "aborted"):
            emit(_turn_end_event(message, []))
            result = AgentLoopResult(
                messages=new_messages,
                surface=list(current_context.messages),
                steps=step,
                termination_reason=(
                    TerminationReason.PROVIDER_ERROR
                    if stop_reason == "error"
                    else TerminationReason.CANCELLED
                ),
                error_detail=_assistant_error_detail(message),
                provider_failure=message.provider_failure,
                metrics=metrics,
                active_provider=state.active_provider,
            )
            emit(_agent_end_event(new_messages, result))
            return result

        # ── 提取工具调用 ──
        tool_calls = [b for b in message.content if isinstance(b, ToolCallContent)]

        if not tool_calls:
            # 模型没有请求工具 → 本轮结束
            emit(_turn_end_event(message, []))

            if config.completion_verifier is not None:
                feedback = await config.completion_verifier(current_context.messages)
                if feedback:
                    _append_messages(
                        current_context,
                        new_messages,
                        [UserMessage(content=feedback)],
                    )
                    continue

            # 原子关闭入口并检查生成期间到达的末轮 steer。若存在，
            # 重新开放入口并让下一次模型调用消费这些消息。
            if finish_steering is not None and (
                config.max_steps is None or step < config.max_steps
            ):
                final_steers = finish_steering()
                if final_steers:
                    _append_messages(current_context, new_messages, final_steers)
                    if reopen_steering is not None:
                        reopen_steering()
                    continue

            return _finish_loop(
                new_messages,
                current_context.messages,
                step,
                metrics,
                state.active_provider,
                emit,
            )

        # ── 工具执行 ──
        tool_started_at = time.perf_counter()
        executed: ExecutedToolBatch = await execute_tool_calls(
            current_context, message, tool_calls, config, signal, emit
        )
        metrics.tool_latencies_ms.append((time.perf_counter() - tool_started_at) * 1000)
        tool_results = executed.results
        _append_tool_results(current_context, new_messages, metrics, tool_results)

        emit(_turn_end_event(message, tool_results))

        # ── 重复工具看门狗 ──
        repeated_watchdog_reason = update_repeated_tool_watchdog(
            state, tool_calls, config, tool_results
        )
        if repeated_watchdog_reason:
            return _finish_loop(
                new_messages,
                current_context.messages,
                step,
                metrics,
                state.active_provider,
                emit,
                termination_reason=TerminationReason.WATCHDOG,
                watchdog_reason=repeated_watchdog_reason,
            )

        # ── 空闲步骤看门狗 ──
        idle_watchdog_reason = update_idle_tool_watchdog(
            state, tool_calls, tool_results, config
        )
        if idle_watchdog_reason:
            return _finish_loop(
                new_messages,
                current_context.messages,
                step,
                metrics,
                state.active_provider,
                emit,
                termination_reason=TerminationReason.WATCHDOG,
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
                    current_context.messages,
                    step,
                    metrics,
                    state.active_provider,
                    emit,
                )

        # 工具已执行，继续内层循环（下一轮工具调用）
        if not executed.terminate:
            continue

    # 步骤耗尽
    result = AgentLoopResult(
        messages=new_messages,
        surface=list(current_context.messages),
        steps=step,
        termination_reason=TerminationReason.STEP_LIMIT,
        metrics=metrics,
        active_provider=state.active_provider,
    )
    emit(_agent_end_event(new_messages, result))
    return result


def _finish_loop(
    new_messages: list[AgentMessage],
    surface: list[AgentMessage],
    step: int,
    metrics: AgentLoopMetrics,
    active_provider: StreamProvider | None,
    emit: Callable[[AgentEvent], None],
    *,
    termination_reason: TerminationReason = TerminationReason.COMPLETED,
    watchdog_reason: str | None = None,
    error_detail: str | None = None,
) -> AgentLoopResult:
    result = AgentLoopResult(
        messages=new_messages,
        surface=list(surface),
        steps=step,
        termination_reason=termination_reason,
        watchdog_reason=watchdog_reason,
        error_detail=error_detail,
        metrics=metrics,
        active_provider=active_provider,
    )
    emit(_agent_end_event(new_messages, result))
    return result


def _append_steering_messages(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    steer_queue: Callable[[], list[AgentMessage]] | None,
) -> None:
    if steer_queue is None:
        return
    msgs = steer_queue()
    if not msgs:
        return
    _append_messages(current_context, new_messages, msgs)


def _append_messages(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    messages: list[AgentMessage],
) -> None:
    """按顺序把运行时消息同时写入上下文和本轮 transcript。"""
    for msg in messages:
        current_context.messages.append(msg)
        new_messages.append(msg)


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


# ── 内层循环 ──


async def _run_inner_loop(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None,
    metrics: AgentLoopMetrics,
    step: int,
    state: _LoopRunState,
) -> tuple[AssistantMessage, str, StreamProvider | None] | None:
    """内层循环：模型调用 → 错误重试 → max_tokens 续写。

    通过 state 对象共享 step_retries / consecutive_continuations 计数器，
    无需外循环再手动同步。
    返回 (message, stop_reason, provider)，或 None 表示应提前退出。
    """
    provider = state.active_provider
    while True:
        if is_cancelled(signal):
            return _cancelled_message(signal), "aborted", provider

        if provider is None:
            msg = AssistantMessage(
                content=[TextContent(text="No provider configured")],
                stop_reason="end_turn",
            )
            emit(_message_start_event(msg))
            emit(_message_end_event(msg))
            return msg, "end_turn", provider

        response = await call_provider(
            context,
            config,
            emit,
            signal,
            metrics,
            provider,
            current_step=step,
        )
        if response is None:
            # 模型流式生成期间被打断：中止在途请求并退出本轮。
            return _cancelled_message(signal), "aborted", provider
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
                context.messages.append(fallback_message)
                return fallback_message, "error", provider
            emit(_message_end_event(message))
            return message, stop_reason, provider

        # ── max_tokens 续写 ──
        if _should_continue_max_tokens(stop_reason, config):
            continuation_count = _update_continuation_count(
                message, state.consecutive_continuations, config
            )
            if continuation_count is None:
                fallback = _continuation_limit_message(config)
                context.messages.append(fallback)
                emit(_message_start_event(fallback))
                emit(_message_end_event(fallback))
                return fallback, "error", provider
            state.consecutive_continuations = continuation_count
            _append_continuation_prompt(context, message)
            continue

        # ── 正常结束 ──
        context.messages.append(message)
        emit(_message_end_event(message))
        state.step_retries = 0
        state.consecutive_continuations = 0
        return message, stop_reason, provider


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
    if config.max_step_retries > 0 or _has_empty_text_response(message):
        msg = AssistantMessage(
            content=[TextContent(text="I encountered an error.")],
            stop_reason="error",
            error_message=_assistant_error_detail(message),
            provider_failure=message.provider_failure,
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
) -> int | None:
    """检测模型是否陷入低产出续写循环。

    续写机制设计：
    - 模型输出因 max_tokens 截断时自动续写
    - 若续写内容少于阈值（默认 500 tokens），计入低产出计数
    - 连续多次低产出（默认 3 次）则终止，防止无限循环
    """
    estimated_tokens = estimate_tokens(
        json.dumps(message.content, ensure_ascii=False, default=str)
    )
    updated = (
        consecutive_continuations + 1
        if estimated_tokens < config.min_continuation_tokens
        else 0
    )
    if updated >= config.max_consecutive_continuations:
        return None
    return updated


def _continuation_limit_message(config: AgentLoopConfig) -> AssistantMessage:
    """构造续写保护触发后的结构化错误消息。"""
    return AssistantMessage(
        content=[
            TextContent(
                text=(
                    "Diminishing Returns: consecutive output token increments "
                    f"below {config.min_continuation_tokens} limit."
                )
            )
        ],
        stop_reason="error",
    )


def _append_continuation_prompt(
    context: AgentContext,
    message: AssistantMessage,
) -> None:
    context.messages.append(message)
    context.messages.append(UserMessage(content="continue"))


def _cancelled_message(signal: CancellationSignal | None) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        stop_reason="aborted",
        error_message=cancel_reason(signal),
    )


def _assistant_error_detail(message: AssistantMessage) -> str | None:
    """提取 provider 或取消返回的具体错误信息。"""
    if message.error_message:
        return message.error_message
    text = " ".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()
    return text or None
