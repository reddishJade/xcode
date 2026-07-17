"""通用工具提示渲染。"""

from __future__ import annotations

from xcode.agent.types import ToolSpec


def build_tool_prompt(registry: tuple[ToolSpec, ...]) -> str:
    """渲染工具摘要及其紧凑参数列表。"""
    lines = []
    for tool in registry:
        snippet = tool.prompt_snippet or tool.description
        if snippet.strip():
            lines.append(f"- {tool.name}: {snippet.strip()}")
        param_lines = compact_tool_params(tool)
        if param_lines:
            lines.extend(param_lines)
    return "\n".join(lines) if lines else "(none)"


def compact_tool_params(tool: ToolSpec) -> list[str]:
    """从 JSON schema 提取供提示词使用的紧凑参数描述。"""
    schema = tool.schema
    if not schema:
        return []
    props = schema.get("properties", {})
    if not props:
        return []
    required = set(schema.get("required", []))
    param_parts: list[str] = []
    for name, prop in props.items():
        typ = prop.get("type", "any")
        if name in required:
            param_parts.append(f"  {name}: {typ}")
        else:
            param_parts.append(f"  {name}?: {typ}")
    if param_parts:
        return ["  Parameters:"] + param_parts
    return []


def build_tool_guidelines(registry: tuple[ToolSpec, ...]) -> str:
    """按注册顺序渲染去重后的工具指南。"""
    guidelines: list[str] = []
    seen: set[str] = set()
    for tool in registry:
        for guideline in tool.prompt_guidelines:
            normalized = guideline.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                guidelines.append(f"- {normalized}")
    return "\n".join(guidelines)
