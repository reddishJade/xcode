from __future__ import annotations

from xcode.ai.types import ToolDefinition
from xcode.harness.skills import ToolSpec


def tool_definition_from_spec(spec: ToolSpec) -> ToolDefinition:
    """从 ToolSpec 提取 LLM 可见 schema。"""
    return ToolDefinition(
        name=spec.name,
        description=spec.description,
        schema=spec.schema or {},
    )


def tool_definitions_from_specs(specs: tuple[ToolSpec, ...]) -> list[ToolDefinition]:
    """批量转换工具 schema。"""
    return [tool_definition_from_spec(spec) for spec in specs]
