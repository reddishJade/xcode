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
from time import perf_counter
from typing import Any

from xcode.ai.events import FinalMessage, TextDelta, ReasoningDelta, ToolCallEvent
from xcode.ai.types import ToolDefinition
from .provider_retry import call_provider_with_retry
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
    first_turn = True
    pending_messages: list[AgentMessage] = []

    # 看门狗状态
    last_tool_signature: str | None = None
    repeated_tool_count: int = 0
    consecutive_idle_steps: int = 0
    consecutive_continuations: int = 0
    step_retries: int = 0

    # Provider 状态
    active_provider = config.provider

    for step in range(1, config.max_steps + 1):
        metrics.steps = step

        # ── 取消检查 ──
        if _is_cancelled(signal):
            result = AgentLoopResult(
                messages=new_messages,
                steps=step,
                metrics=metrics,
                active_provider=active_provider,
            )
            emit(_agent_end_event(new_messages, result))
            return result

        # ── 处理 steer 队列 ──
        if config.get_steering_messages:
            steer_msgs = config.get_steering_messages()
            if steer_msgs:
                for msg in steer_msgs:
                    current_context.messages.append(msg)
                    new_messages.append(msg)

        # ── 发出 turn 事件 ──
        if not first_turn:
            emit(_turn_start_event())
        else:
            first_turn = False

        # ── 处理 pending messages ──
        if pending_messages:
            for msg in pending_messages:
                emit(_message_start_event(msg))
                emit(_message_end_event(msg))
                current_context.messages.append(msg)
                new_messages.append(msg)
            pending_messages = []

        # ── 压缩检查 ──
        if config.should_compact and config.compact:
            if config.should_compact(current_context.messages):
                current_context.messages = config.compact(current_context.messages)

        # ── 内层循环：模型调用 + 重试 + max_tokens ──
        ctx_len_before = len(current_context.messages)
        inner_result = await _run_inner_loop(
            current_context,
            config,
            emit,
            signal,
            metrics,
            step,
            active_provider,
            step_retries,
            consecutive_continuations,
        )

        if inner_result is None:
            # 内层循环决定提前退出（错误耗尽 or 递减收益）
            result = AgentLoopResult(
                messages=new_messages,
                steps=step,
                metrics=metrics,
                active_provider=active_provider,
            )
            emit(_agent_end_event(new_messages, result))
            return result

        message, stop_reason, new_provider = inner_result
        active_provider = new_provider

        # 同步内层循环添加的中间消息（如 max_tokens 续写的 "continue"）
        for msg in current_context.messages[ctx_len_before:-1]:
            new_messages.append(msg)

        # ── 更新步骤重试状态 ──
        if stop_reason == "error":
            step_retries += 1
        else:
            step_retries = 0
            consecutive_continuations = 0

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
                active_provider=active_provider,
            )

        # ── 提取工具调用 ──
        tool_calls = [b for b in message.content if isinstance(b, ToolCallContent)]

        if not tool_calls:
            # 模型没有请求工具 → 本轮结束
            emit(_turn_end_event(message, []))

            # 检查 follow-up 队列
            if config.get_follow_up_messages:
                follow_up = config.get_follow_up_messages()
                if follow_up:
                    pending_messages = follow_up
                    continue
            result = AgentLoopResult(
                messages=new_messages,
                steps=step,
                metrics=metrics,
                active_provider=active_provider,
            )
            emit(_agent_end_event(new_messages, result))
            return result

        # ── 工具执行 ──
        executed: ExecutedToolBatch = await execute_tool_calls(
            current_context, message, tool_calls, config, signal, emit
        )
        tool_results = executed.results
        for tr in tool_results:
            current_context.messages.append(tr)
            new_messages.append(tr)
            metrics.tool_calls += 1

        emit(_turn_end_event(message, tool_results))

        # ── 重复工具看门狗 ──
        sig = _tool_signature(tool_calls)
        if sig == last_tool_signature:
            repeated_tool_count += 1
        else:
            repeated_tool_count = 0
            last_tool_signature = sig

        if (
            config.watchdog_repeated_tool_limit > 0
            and repeated_tool_count >= config.watchdog_repeated_tool_limit
        ):
            reason = f"watchdog stopped repeated tool call: {tool_calls[0].name}"
            result = AgentLoopResult(
                messages=new_messages,
                steps=step,
                stopped_by_watchdog=True,
                watchdog_reason=reason,
                metrics=metrics,
                active_provider=active_provider,
            )
            emit(_agent_end_event(new_messages, result))
            return result

        # ── 空闲步骤看门狗 ──
        is_productive = config.is_tool_productive or _is_tool_productive_default
        if is_productive(tool_calls, tool_results):
            consecutive_idle_steps = 0
        else:
            consecutive_idle_steps += 1

        if (
            config.max_consecutive_idle_steps > 0
            and consecutive_idle_steps >= config.max_consecutive_idle_steps
        ):
            reason = (
                f"Watchdog triggered: {consecutive_idle_steps} consecutive steps "
                f"without productive tool calls."
            )
            result = AgentLoopResult(
                messages=new_messages,
                steps=step,
                stopped_by_watchdog=True,
                watchdog_reason=reason,
                metrics=metrics,
                active_provider=active_provider,
            )
            emit(_agent_end_event(new_messages, result))
            return result

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
                result = AgentLoopResult(
                    messages=new_messages,
                    steps=step,
                    metrics=metrics,
                    active_provider=active_provider,
                )
                emit(_agent_end_event(new_messages, result))
                return result

        # ── 处理 follow-up 队列 ──
        if config.get_follow_up_messages:
            follow_up = config.get_follow_up_messages()
            if follow_up:
                pending_messages = follow_up
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
        active_provider=active_provider,
    )
    emit(_agent_end_event(new_messages, result))
    return result


# ── 内层循环 ──


async def _run_inner_loop(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None,
    metrics: AgentLoopMetrics,
    step: int,
    provider: Any,
    step_retries: int,
    consecutive_continuations: int,
) -> tuple[AssistantMessage, str, Any] | None:
    """内层循环：模型调用 → 错误重试 → max_tokens 续写。

    返回 (message, stop_reason, provider)，或 None 表示应提前退出。
    """
    while True:
        if _is_cancelled(signal):
            return _cancelled_message(signal), "aborted", provider

        # ── 上下文变换 ──
        messages = context.messages
        if config.transform_context:
            messages = config.transform_context(messages, signal)

        # ── 转换为 LLM 消息格式 ──
        convert_fn = config.convert_to_llm or (lambda msgs: [])
        llm_messages = convert_fn(messages)

        # ── 调用 provider（含重试）──
        tool_definitions = _tools_to_definitions(context.tools)
        started = perf_counter()

        if provider is None:
            msg = AssistantMessage(
                content=[TextContent(text="No provider configured")],
                stop_reason="end_turn",
            )
            emit(_message_start_event(msg))
            emit(_message_end_event(msg))
            return msg, "end_turn", provider

        events = await call_provider_with_retry(
            provider,
            llm_messages,
            tool_definitions,
            max_retries=config.max_step_retries,
            backoff_base=config.retry_backoff_base,
            options=config.options,
        )

        elapsed = round((perf_counter() - started) * 1000, 3)
        metrics.model_latencies_ms.append(elapsed)

        # ── 组装响应 ──
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_found: list[ToolCallContent] = []
        stop_reason = "end_turn"

        for event in events:
            if isinstance(event, TextDelta):
                text_parts.append(event.chunk)
                emit(
                    _message_update_event(
                        AssistantMessage(
                            content=[TextContent(text="".join(text_parts))],
                        )
                    )
                )
            elif isinstance(event, ReasoningDelta):
                reasoning_parts.append(event.chunk)
            elif isinstance(event, ToolCallEvent):
                for call in event.calls:
                    tool_calls_found.append(
                        ToolCallContent(
                            id=call.id,
                            name=call.name,
                            arguments=dict(call.input),
                        )
                    )
            # FinalMessage 设置 stop_reason
            if isinstance(event, FinalMessage):
                stop_reason = getattr(event, "stop_reason", "end_turn") or "end_turn"

        final_text = "".join(text_parts)
        content_blocks: list[ContentBlock] = [TextContent(text=final_text)]
        content_blocks.extend(tool_calls_found)

        valid_stop_reasons = (
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "error",
            "aborted",
        )
        final_stop_reason = (
            stop_reason if stop_reason in valid_stop_reasons else "end_turn"
        )
        message = AssistantMessage(
            content=content_blocks,
            reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            stop_reason=final_stop_reason,  # type: ignore[arg-type]
        )

        # ── 检查是否为 FinalMessage 的错误 ──
        if stop_reason == "error":
            step_retries += 1
            if step_retries <= config.max_step_retries:
                delay = config.retry_backoff_base * (2 ** (step_retries - 1))
                await asyncio.sleep(delay)
                continue
            # 重试耗尽
            if not content_blocks or (
                len(content_blocks) == 1
                and isinstance(content_blocks[0], TextContent)
                and not content_blocks[0].text
            ):
                msg = AssistantMessage(
                    content=[TextContent(text="I encountered an error.")],
                    stop_reason="error",
                )
                emit(_message_start_event(msg))
                emit(_message_end_event(msg))
                return None
            emit(_message_end_event(message))
            return message, stop_reason, provider

        # ── max_tokens 续写 ──
        if stop_reason == "max_tokens" and config.max_tokens_continuation:
            inc = _estimate_text_tokens(
                json.dumps(content_blocks, ensure_ascii=False, default=str)
            )
            consecutive_continuations = (
                consecutive_continuations + 1
                if inc < config.min_continuation_tokens
                else 0
            )
            if consecutive_continuations >= config.max_consecutive_continuations:
                raise RuntimeError(
                    "Diminishing Returns: consecutive output token increments "
                    f"below {config.min_continuation_tokens} limit."
                )
            # 将当前消息加入 context，追加 "continue"，重新循环
            context.messages.append(message)
            context.messages.append(UserMessage(content="continue"))
            continue

        # ── 正常结束 ──
        context.messages.append(message)
        emit(_message_end_event(message))
        step_retries = 0
        consecutive_continuations = 0
        return message, stop_reason, provider


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
