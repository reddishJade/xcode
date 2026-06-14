"""工具调用分批调度逻辑。

从 tool_execution.py 提取，专注于按 execution_mode 将工具调用分批。
"""

from __future__ import annotations

from .config import AgentContext
from .types import ToolCallContent


def partition_tool_calls_for_execution(
    current_context: AgentContext,
    tool_calls: list[ToolCallContent],
) -> list[list[ToolCallContent]]:
    """按工具执行模式将连续并发调用分批。"""
    batches: list[list[ToolCallContent]] = []
    parallel_batch: list[ToolCallContent] = []
    for tool_call in tool_calls:
        if _tool_execution_mode(current_context, tool_call) == "parallel":
            parallel_batch.append(tool_call)
            continue
        if parallel_batch:
            batches.append(parallel_batch)
            parallel_batch = []
        batches.append([tool_call])
    if parallel_batch:
        batches.append(parallel_batch)
    return batches


def _tool_execution_mode(
    current_context: AgentContext,
    tool_call: ToolCallContent,
) -> str:
    """返回工具的执行模式（parallel 或 sequential）。

    默认 sequential 的设计原因：
    - 保守策略：避免副作用工具并发执行导致状态不一致
    - 例如：先 write_file 再 read_file，若并行则 read 可能读到旧内容
    - 工具可通过 metadata 显式声明 execution_mode="parallel" 允许并发
    """
    for tool in current_context.tools or []:
        if tool.name == tool_call.name:
            return tool.execution_mode or "sequential"
    return "sequential"
