"""Provider 交互逻辑。"""

from __future__ import annotations

import json
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
from xcode.ai.providers.codec import provider_function_name
from xcode.ai.providers.protocol import StreamProvider
from xcode.ai.types import ToolDefinition
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.context_assembly import (
    ContextAssemblyInput,
    ContextBlock,
    ContextBlockTarget,
    normalize_context_blocks,
    sanitize_assembled_messages,
)
from xcode.agent.context_collector import ContextCollectionInput
from xcode.agent.context_sanitizer import demote_embedded_memory_sections
from xcode.agent.events import (
    AgentEvent,
    MessageUpdateEvent,
    ThinkingUpdateEvent,
)
from xcode.agent.messages import AssistantMessage
from xcode.agent.protocols import AgentTool, CancellationSignal, ContentBlock
from xcode.agent.results import AgentLoopMetrics
from xcode.agent.types import TextContent, ToolCallContent


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
    """收集、规范化、组装并最终发送当前 provider 请求。"""
    original_messages = list(context.messages)
    messages = list(context.messages)
    blocks: list[ContextBlock] = []

    if config.context_collectors and config.context_assembler:
        collect_input = ContextCollectionInput(
            system_prompt=context.system_prompt,
            messages=messages,
            tools=list(context.tools or []),
            current_step=current_step,
        )
        blocks = normalize_context_blocks(config.context_collectors.collect(collect_input))

    trusted_system_blocks = [
        block for block in blocks if block.target is ContextBlockTarget.SYSTEM
    ]

    if config.context_assembler:
        assembly_input = ContextAssemblyInput(
            system_prompt=context.system_prompt,
            messages=messages,
            tools=list(context.tools or []),
            context_blocks=blocks,
            current_step=current_step,
        )
        assembly_result = config.context_assembler.assemble(assembly_input)
        messages = sanitize_assembled_messages(
            original_messages=original_messages,
            assembled_messages=assembly_result.messages,
            trusted_system_blocks=trusted_system_blocks,
        )

    if config.transform_context:
        messages = config.transform_context(messages, signal)

    # transform_context 是旧扩展点，仍必须经过同一 provider 边界复核。
    messages = sanitize_assembled_messages(
        original_messages=original_messages,
        assembled_messages=messages,
        trusted_system_blocks=trusted_system_blocks,
    )
    messages = demote_embedded_memory_sections(messages)

    convert_fn = config.convert_to_llm or (lambda msgs: [])
    llm_messages = convert_fn(messages)
    tool_definitions = _tools_to_definitions(context.tools)
    if config.before_provider_request:
        config.before_provider_request(llm_messages, tool_definitions)

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
    provider: StreamProvider,
    llm_messages: list[Message],
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
    except Exception as exc:
        return [FinalMessage(content=f"Provider error: {exc}", stop_reason="error")]


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
            AssistantMessage(content=[TextContent(text="".join(text_parts))])
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


def _tools_to_definitions(tools: list[AgentTool] | None) -> list[ToolDefinition]:
    if not tools:
        return []
    result: list[ToolDefinition] = []
    for tool in tools:
        description = tool.description
        examples = getattr(tool, "examples", [])
        if examples:
            example_lines = ["\n", "Examples:"]
            for example in examples:
                example_lines.append(
                    f"  - {example.get('name', '')}: "
                    f"input={json.dumps(example.get('input', {}), ensure_ascii=False)}, "
                    f'output="{example.get("output", "")}"'
                )
            description += "\n".join(example_lines)
        builtin = getattr(tool, "builtin", None)
        provider_name = provider_function_name(tool.name)
        result.append(
            ToolDefinition(
                name=provider_name,
                description=description,
                parameters=dict(tool.parameters),
                builtin=builtin if isinstance(builtin, dict) else None,
            )
        )
    return result
