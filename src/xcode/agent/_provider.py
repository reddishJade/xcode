"""Provider 交互逻辑。

从 agent_loop.py 提取的 provider 调用、事件收集和消息组装逻辑。

**提取的设计原因**：
- 关注点分离：agent_loop.py 专注于循环编排，_provider.py 专注于 LLM 交互
- 测试隔离：provider 交互逻辑可以独立测试
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Awaitable, cast

from xcode.ai.events import (
    FinalMessage,
    Message,
    ProviderEvent,
    ReasoningDelta,
    StopReason,
    TextDelta,
    ToolCallEvent,
    UsageUpdate,
)
from xcode.ai.providers.base import StreamProvider
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.agent.types import (
    CancellationSignal,
    ContentBlock,
    TextContent,
    ToolCallContent,
)
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.results import AgentLoopMetrics
from xcode.agent.events import (
    AgentEvent,
    MessageUpdateEvent,
    ThinkingUpdateEvent,
)
from xcode.agent.messages import AssistantMessage


@dataclass
class _ProviderResponse:
    message: AssistantMessage
    stop_reason: StopReason


async def call_provider(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None,
    metrics: AgentLoopMetrics,
    provider: StreamProvider,
    current_step: int = 0,
) -> _ProviderResponse | None:
    """调用 provider；若流式生成期间被打断则返回 None。"""
    assembly = config.request_assembler.assemble(
        context,
        current_step=current_step,
        options=config.options,
    )
    if config.before_provider_request:
        config.before_provider_request(assembly)

    started = perf_counter()
    events = await _collect_provider_events(
        provider,
        list(assembly.wire_messages),
        list(assembly.tools),
        assembly.options,
        emit,
        signal,
    )
    elapsed = round((perf_counter() - started) * 1000, 3)
    metrics.model_latencies_ms.append(elapsed)
    if events is None:
        return None
    return _provider_events_to_response(events, metrics, lambda _event: None)


def _is_cancelled(signal: CancellationSignal | None) -> bool:
    return signal is not None and signal.is_cancelled()


def _abort_inflight_stream(provider: StreamProvider) -> None:
    """尽力中止在途 HTTP 流：关闭底层连接使阻塞读取立刻失败。"""
    abort = getattr(provider, "abort_active_stream", None)
    if not callable(abort):
        return
    try:
        abort()
    except Exception:
        pass


async def _aclose_stream(stream_iter: AsyncIterator[ProviderEvent]) -> None:
    """尽快关闭异步流迭代器，让底层生成器的清理逻辑及时执行。"""
    aclose = getattr(stream_iter, "aclose", None)
    if not callable(aclose):
        return
    try:
        await cast(Awaitable[None], aclose())
    except Exception:
        pass


async def _collect_provider_events(
    provider: StreamProvider,
    llm_messages: list[Message],
    tool_definitions: list[ToolDefinition],
    options: StreamOptions | None,
    emit: Callable[[AgentEvent], None],
    signal: CancellationSignal | None = None,
) -> list[ProviderEvent] | None:
    """逐事件收集 provider 流；取消时中止在途请求并返回 None。"""
    events: list[ProviderEvent] = []
    text_parts: list[str] = []
    try:
        kwargs = {}
        if options is not None:
            kwargs["options"] = options
        stream_iter = provider.stream(llm_messages, tool_definitions, **kwargs)
        async for event in stream_iter:
            if _is_cancelled(signal):
                _abort_inflight_stream(provider)
                await _aclose_stream(stream_iter)
                return None
            events.append(event)
            if isinstance(event, TextDelta):
                _append_text_delta(text_parts, event, emit)
                await asyncio.sleep(0)
            elif isinstance(event, ReasoningDelta):
                emit(ThinkingUpdateEvent(reasoning_content=event.chunk))
                await asyncio.sleep(0)
        return events
    except Exception as e:
        if _is_cancelled(signal):
            # 打断触发的连接关闭会使阻塞读取抛出异常，属预期路径。
            return None
        events.append(FinalMessage(content=f"Provider error: {e}", stop_reason="error"))
        return events


def _provider_events_to_response(
    events: list[ProviderEvent],
    metrics: AgentLoopMetrics,
    emit: Callable[[AgentEvent], None],
) -> _ProviderResponse:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_found: list[ToolCallContent] = []
    stop_reason: StopReason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    has_usage = False
    final_content: str | None = None

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
            input_tokens += event.input_tokens
            output_tokens += event.output_tokens
            has_usage = True
        if isinstance(event, FinalMessage):
            stop_reason = event.stop_reason or "end_turn"
            if event.content:
                final_content = event.content

    if final_content and not text_parts:
        text_parts.append(final_content)

    content_blocks: list[ContentBlock] = [TextContent(text="".join(text_parts))]
    content_blocks.extend(tool_calls_found)
    usage = None
    if has_usage:
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    return _ProviderResponse(
        message=AssistantMessage(
            content=content_blocks,
            reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            stop_reason=stop_reason,
            error_message=final_content if stop_reason == "error" else None,
            usage=usage,
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


def _message_update_event(message: AssistantMessage) -> MessageUpdateEvent:
    return MessageUpdateEvent(message=message)


def _tool_call_content_blocks(event: ToolCallEvent) -> list[ToolCallContent]:
    return [
        ToolCallContent(
            id=call.id,
            name=call.name,
            arguments=dict(call.input),
        )
        for call in event.calls
    ]
