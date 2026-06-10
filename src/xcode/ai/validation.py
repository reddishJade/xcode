"""工具参数校验层。

基于 jsonschema 的标准 JSON Schema 校验。
与 ToolDefinition.parameters 配合使用。
"""

from __future__ import annotations

from typing import Any

import jsonschema

from .types import ToolDefinition


class ToolValidationError(ValueError):
    """工具参数校验失败时抛出。"""


def validate_tool_call(
    tools: list[ToolDefinition],
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """校验工具参数。

    参数:
        tools: 已注册的工具定义列表
        name: 被调用的工具名
        arguments: 模型生成的参数 dict

    返回:
        校验后的参数 dict（与输入相同）

    抛出:
        ToolValidationError: 校验失败时
    """
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        msg = f"Unknown tool: {name}. Available: {[t.name for t in tools]}"
        raise ToolValidationError(msg)

    if not tool.parameters:
        return arguments

    try:
        jsonschema.validate(
            instance=arguments,
            schema=tool.parameters,
        )
    except jsonschema.ValidationError as exc:
        path = (
            ".".join(str(p) for p in exc.absolute_path) if exc.absolute_path else name
        )
        raise ToolValidationError(f"{path}: {exc.message}") from exc

    return arguments
