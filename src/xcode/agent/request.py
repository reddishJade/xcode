"""Provider 请求的单一组装边界。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from xcode.ai.events import Message
from xcode.ai.providers._codec import provider_function_name
from xcode.ai.types import StreamOptions, ToolDefinition

from ._codec import convert_to_llm
from ._compaction import estimate_tokens
from ._hygiene import apply_request_hygiene
from .context import (
    ContextAssembler,
    ContextAssemblyInput,
    ContextBlock,
    ContextCollectionInput,
    ContextCollectorSource,
    DefaultContextAssembler,
)
from .messages import AgentMessage
from .types import AgentTool, materialize_json_mapping

if TYPE_CHECKING:
    from .config import AgentContext

type MessageConverter = Callable[[list[AgentMessage]], list[Message]]


@dataclass(frozen=True)
class RequestHygiene:
    """请求组装阶段唯一允许的确定性消息裁剪策略。"""

    enabled: bool = True
    max_tool_result_bytes: int = 8000
    max_tool_arg_length: int = 1000
    keep_head_lines: int = 50
    keep_tail_lines: int = 50

    def apply(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        if not self.enabled:
            return list(messages)
        return apply_request_hygiene(
            messages,
            max_tool_result_bytes=self.max_tool_result_bytes,
            max_tool_arg_length=self.max_tool_arg_length,
            keep_head_lines=self.keep_head_lines,
            keep_tail_lines=self.keep_tail_lines,
        )


@dataclass(frozen=True)
class RequestContextTrace:
    """一项动态上下文对本次请求的可审计决议。"""

    source: str
    target: str
    block_id: str
    included: bool
    token_count: int
    content_sha256: str
    provenance: str = ""
    truncated: bool = False
    truncation_reason: str | None = None
    scope: str = "project"


@dataclass(frozen=True)
class RequestAssembly:
    """传给 provider 与审计 hook 的同一份请求快照。"""

    messages: tuple[AgentMessage, ...]
    wire_messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    context_trace: tuple[RequestContextTrace, ...]
    current_step: int
    hygiene_applied: bool
    estimated_tokens: int = 0
    token_budget: int = 0
    budget_remaining: int = 0
    options: StreamOptions | None = None


class RequestAssembler(Protocol):
    """从 agent 当前投影生成一次完整 provider 请求。"""

    def assemble(
        self,
        context: AgentContext,
        *,
        current_step: int,
        options: StreamOptions | None,
    ) -> RequestAssembly: ...


@dataclass(frozen=True)
class DefaultRequestAssembler:
    """按固定阶段收集、注入、裁剪并编码模型输入。"""

    converter: MessageConverter = convert_to_llm
    context_collectors: ContextCollectorSource | None = None
    context_assembler: ContextAssembler = field(default_factory=DefaultContextAssembler)
    hygiene: RequestHygiene = field(default_factory=RequestHygiene)

    def assemble(
        self,
        context: AgentContext,
        *,
        current_step: int,
        options: StreamOptions | None,
    ) -> RequestAssembly:
        context.context_state.sync_request_prefix(context.request_prefix)
        legacy_blocks, world_blocks = self._collect(
            context,
            list(context.messages),
            current_step,
        )
        context.context_state.append_blocks(world_blocks)
        base_messages = [
            *context.context_state.persistent_messages,
            *context.messages,
        ]
        result = self.context_assembler.assemble(
            ContextAssemblyInput(
                system_prompt=context.system_prompt,
                messages=base_messages,
                tools=list(context.tools),
                context_blocks=legacy_blocks,
                current_step=current_step,
                token_budget=context.request_token_budget,
            )
        )
        messages = self.hygiene.apply(result.messages)
        wire_messages = self.converter(messages)
        tool_definitions = _tools_to_definitions(context.tools)
        return RequestAssembly(
            messages=tuple(messages),
            wire_messages=tuple(wire_messages),
            tools=tuple(tool_definitions),
            context_trace=(
                _context_trace(
                    [*world_blocks, *result.blocks_used],
                    result.blocks_dropped,
                )
                + _tool_trace(tool_definitions)
            ),
            current_step=current_step,
            hygiene_applied=self.hygiene.enabled,
            estimated_tokens=result.total_tokens,
            token_budget=result.token_budget,
            budget_remaining=result.budget_remaining,
            options=options,
        )

    def _collect(
        self,
        context: AgentContext,
        messages: list[AgentMessage],
        current_step: int,
    ) -> tuple[list[ContextBlock], list[ContextBlock]]:
        if self.context_collectors is None:
            return [], []
        collection_input = ContextCollectionInput(
            system_prompt=context.system_prompt,
            messages=messages,
            tools=list(context.tools),
            current_step=current_step,
            project_root=context.project_root,
            cwd=context.cwd,
            state=dict(context.state),
        )
        legacy_blocks = self.context_collectors.collect(collection_input)
        collect_sections = getattr(self.context_collectors, "collect_sections", None)
        world_blocks: list[ContextBlock] = []
        if callable(collect_sections):
            collected = collect_sections(
                collection_input,
                context.context_state.world_state,
            )
            if isinstance(collected, list):
                world_blocks = [
                    block for block in collected if isinstance(block, ContextBlock)
                ]
        return legacy_blocks, world_blocks


def _context_trace(
    used: list[ContextBlock],
    dropped: list[ContextBlock],
) -> tuple[RequestContextTrace, ...]:
    return tuple(
        _trace(block, included)
        for included, blocks in ((True, used), (False, dropped))
        for block in blocks
    )


def _trace(block: ContextBlock, included: bool) -> RequestContextTrace:
    return RequestContextTrace(
        source=block.source.value,
        target=block.target.value,
        block_id=block.block_id,
        included=included,
        token_count=block.get_token_count(),
        content_sha256=hashlib.sha256(block.content.encode("utf-8")).hexdigest(),
        provenance=block.provenance,
        truncated=block.truncated,
        truncation_reason=block.truncation_reason,
        scope=block.scope,
    )


def _tool_trace(tools: list[ToolDefinition]) -> tuple[RequestContextTrace, ...]:
    traces: list[RequestContextTrace] = []
    for tool in tools:
        payload = json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "builtin": tool.builtin,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        traces.append(
            RequestContextTrace(
                source="tool",
                target="tool_definition",
                block_id=tool.name,
                included=True,
                token_count=estimate_tokens(payload),
                content_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                provenance=f"tool:{tool.name}",
                scope="runtime",
            )
        )
    return tuple(traces)


def _tools_to_definitions(tools: list[AgentTool]) -> list[ToolDefinition]:
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
        result.append(
            ToolDefinition(
                name=provider_function_name(tool.name),
                description=description,
                parameters=materialize_json_mapping(tool.parameters),
                builtin=builtin if isinstance(builtin, dict) else None,
            )
        )
    return result
