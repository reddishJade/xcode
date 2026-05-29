from __future__ import annotations

from typing import Any

from xcode.agent.types import AgentTool


def wrap_tool(tool: Any) -> AgentTool[Any]:
    """将任意兼容对象转为 AgentTool。

    要求 tool 对象具有 name, label, description, parameters 属性
    和 async execute(tool_call_id, params, signal, on_update) 方法。
    """
    if isinstance(tool, AgentTool):  # type: ignore
        return tool
    return tool  # type: ignore


def wrap_tools(tools: list[Any]) -> list[AgentTool[Any]]:
    return [wrap_tool(t) for t in tools]
