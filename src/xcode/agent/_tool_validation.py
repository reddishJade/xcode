"""工具参数校验逻辑。

从 tool_execution.py 提取，专注于 JSON Schema 参数校验。
"""

from __future__ import annotations

import jsonschema

from xcode.agent.types import ToolArguments, ToolCallContent
from .protocols import AgentTool


def validate_tool_arguments(
    tool: AgentTool,
    tool_call: ToolCallContent,
    args: ToolArguments,
) -> str | None:
    """按工具 JSON schema 校验模型生成的参数。"""
    try:
        schema = dict(tool.parameters)
    except Exception as exc:
        return f"tool schema error for {tool_call.name}: {exc}"
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        path = (
            ".".join(str(part) for part in exc.absolute_path)
            if exc.absolute_path
            else tool_call.name
        )
        return f"tool argument schema error: {path}: {exc.message}"
    return None
