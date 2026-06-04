from __future__ import annotations

from typing import Any

from .types import ToolDefinition

"""工具参数校验层。

基于 JSON Schema 的轻量校验，无外部依赖。
与 ToolDefinition.schema 配合使用。
"""


class ToolValidationError(ValueError):
    """工具参数校验失败时抛出。"""


_TYPE_NAMES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _validate_value(
    value: object,
    schema: dict[str, Any],
    path: str = "",
) -> list[str]:
    errors: list[str] = []

    if "type" in schema:
        expected = schema["type"]
        actual = _TYPE_NAMES.get(type(value), type(value).__name__)

        if expected == "integer" and actual == "number" and isinstance(value, bool):
            errors.append(f"{path}: expected integer, got boolean")
        elif expected == "integer" and actual == "number" and isinstance(value, float):
            if value != int(value):
                errors.append(f"{path}: expected integer, got float {value}")
        elif expected == "integer" and actual == "number":
            pass
        elif expected != actual and not (expected == "number" and actual == "integer"):
            errors.append(f"{path}: expected {expected}, got {actual}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in {schema['enum']}")

    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        additional = schema.get("additionalProperties", True)

        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")

        for key, val in value.items():
            sub_path = f"{path}.{key}" if path else key
            if key in props:
                errors.extend(_validate_value(val, props[key], sub_path))
            elif not additional:
                errors.append(f"{sub_path}: unexpected field")

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(list(value)):
            errors.extend(_validate_value(item, schema["items"], f"{path}[{i}]"))

    return errors


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

    if not tool.schema:
        return arguments

    errors = _validate_value(arguments, tool.schema, name)
    if errors:
        raise ToolValidationError("; ".join(errors))

    return arguments
