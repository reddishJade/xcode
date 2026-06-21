"""ToolSpec ↔ AgentTool 适配器。

将 harness 层的 ToolSpec 适配为 agent 层的 AgentTool protocol（正向），
以及将 AgentTool 反向适配为 ToolSpec（反向），
使 StructuredAgent 可以将 ToolSpec 注册表传给 agent 核心循环，
也允许第三方 AgentTool 注册到 harness 层。
"""

from __future__ import annotations

import asyncio

from ...agent.protocols import (
    AgentTool,
    AgentToolResult,
    CancellationSignal,
    ToolExecutionMode,
    ToolResultContentBlock,
    ToolUpdateCallback,
)
from xcode.agent.types import (
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ToolArguments,
)
from ..skills import (
    AGENT_CONTENT_BLOCKS_METADATA_KEY,
    ToolSpec,
)
from ..observability import redact_text


class ToolSpecAdapter:
    """将 harness ToolSpec 适配为 agent AgentTool protocol。

    依赖方向：harness -> agent（正确）。
    此类在 harness/ 层，实现 agent/ 层定义的 protocol。
    """

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def label(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> dict[str, object]:
        if self._spec.schema is None:
            raise ValueError(f"tool {self._spec.name} must define a JSON schema")
        return self._spec.schema

    @property
    def examples(self) -> list[dict[str, object]]:
        return list(self._spec.examples)

    @property
    def builtin(self) -> dict[str, object] | None:
        return self._spec.builtin

    @property
    def execution_mode(self) -> ToolExecutionMode | None:
        if self._spec.execution_mode is not None:
            return self._spec.execution_mode
        if self._spec.read_only and self._spec.concurrency_safe:
            return "parallel"
        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        params: ToolArguments,
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        tool_input = _tool_input_from_arguments(params)
        content = await asyncio.to_thread(self._spec.handler, tool_input)
        metadata = getattr(content, "metadata", None)
        return AgentToolResult(
            content=_tool_result_content(content, tool_call_id),
            details=metadata if isinstance(metadata, dict) else None,
            is_error=bool(getattr(content, "is_error", False)),
        )


def adapt_tool_specs(specs: tuple[ToolSpec, ...]) -> list[ToolSpecAdapter]:
    """批量将 ToolSpec 适配为 AgentTool。

    权限门控由 ToolGate 在 before_tool_call 钩子中处理，不在本适配器中执行。
    """
    missing_schema = [spec.name for spec in specs if spec.schema is None]
    if missing_schema:
        names = ", ".join(sorted(missing_schema))
        raise ValueError(f"tools must define JSON schemas: {names}")
    return [ToolSpecAdapter(spec) for spec in specs]


def _tool_result_content(
    raw_content: str,
    tool_call_id: str,
) -> list[ToolResultContentBlock]:
    """从 ToolOutput 元数据中保留结构化结果块。"""
    content: list[ToolResultContentBlock] = [
        TextContent(text=redact_text(str(raw_content)))
    ]
    metadata = getattr(raw_content, "metadata", None)
    if not isinstance(metadata, dict):
        return content

    raw_blocks = metadata.get(AGENT_CONTENT_BLOCKS_METADATA_KEY)
    if not isinstance(raw_blocks, list):
        return content

    content.extend(_structured_content_blocks(raw_blocks, tool_call_id))
    return content


def _structured_content_blocks(
    raw_blocks: list[object],
    tool_call_id: str,
) -> list[ToolResultContentBlock]:
    """过滤并规范化可传给 agent loop 的结构化内容块。"""
    blocks: list[ToolResultContentBlock] = []
    for block in raw_blocks:
        if isinstance(block, ShellCallOutputContent):
            blocks.append(_redacted_shell_call_output(block, tool_call_id))
        elif isinstance(block, ImageContent | FileContent):
            blocks.append(block)
    return blocks


def _redacted_shell_call_output(
    block: ShellCallOutputContent,
    tool_call_id: str,
) -> ShellCallOutputContent:
    """对 shell stdout/stderr 执行与普通工具输出一致的脱敏。"""
    return ShellCallOutputContent(
        call_id=block.call_id or tool_call_id,
        output=[_redacted_shell_output_item(item) for item in block.output],
        max_output_length=block.max_output_length,
    )


def _redacted_shell_output_item(item: dict[str, object]) -> dict[str, object]:
    redacted = dict(item)
    stdout = redacted.get("stdout")
    stderr = redacted.get("stderr")
    if isinstance(stdout, str):
        redacted["stdout"] = redact_text(stdout)
    if isinstance(stderr, str):
        redacted["stderr"] = redact_text(stderr)
    return redacted


def create_tool_spec_from_agent_tool(tool: AgentTool) -> ToolSpec:
    """将 AgentTool 反向适配为 ToolSpec。

    在测试或第三方集成中，调用方提供了 AgentTool 实例而 harness
    需要 ToolSpec 时使用。
    """
    import asyncio

    async def execute_async(data):
        return await tool.execute("", data, None)

    def sync_handler(data):
        result = asyncio.run(execute_async(data))
        return "".join(c.text for c in result.content if isinstance(c, TextContent))

    return ToolSpec(
        name=tool.name,
        description=tool.description,
        input_hint=(
            f"JSON: {tool.parameters.get('properties', {})!r}"
            if tool.parameters
            else "{}"
        ),
        handler=sync_handler,
        schema=dict(tool.parameters),
        execution_mode=tool.execution_mode,
        examples=list(tool.examples),
    )


def _tool_input_from_arguments(arguments: ToolArguments) -> dict[str, object]:
    return dict(arguments)
