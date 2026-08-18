"""Provider 交互逻辑。

从 agent_loop.py 提取的 provider 调用、事件收集和消息组装逻辑。

**提取的设计原因**：
- 关注点分离：agent_loop.py 专注于循环编排，_provider.py 专注于 LLM 交互
- 测试隔离：provider 交互逻辑可以独立测试
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

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
) -> _ProviderResponse:
    del signal
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
    )
    elapsed = round((perf_counter() - started) * 1000, 3)
    metrics.model_latencies_ms.append(elapsed)
    return _provider_events_to_response(events, metrics, lambda _event: None)


async def _collect_provider_events(
    provider: StreamProvider,
    llm_messages: list[Message],
    tool_definitions: list[ToolDefinition],
    options: StreamOptions | None,
    emit: Callable[[AgentEvent], None],
) -> list[ProviderEvent]:
    events: list[ProviderEvent] = []
    text_parts: list[str] = []
    try:
        kwargs = {}
        if options is not None:
            kwargs["options"] = options
        async for event in provider.stream(llm_messages, tool_definitions, **kwargs):
            events.append(event)
            if isinstance(event, TextDelta):
                _append_text_delta(text_parts, event, emit)
                await asyncio.sleep(0)
            elif isinstance(event, ReasoningDelta):
                emit(ThinkingUpdateEvent(reasoning_content=event.chunk))
                await asyncio.sleep(0)
        return events
    except Exception as e:
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
