"""重复工具调用检测和抑制增强。

提供带文件变更感知的工具调用签名生成和重复检测。
"""

from __future__ import annotations

import json

from xcode.agent.config import AgentLoopConfig, _LoopRunState
from xcode.agent.messages import ToolResultMessage
from xcode.agent.types import ToolCallContent


def tool_call_signature(call: ToolCallContent) -> str:
    """生成单个工具调用的规范化签名。

    签名生成规则：
    - 工具名 + 参数 JSON（排序键）
    - 用于检测重复调用
    """
    args_str = json.dumps(call.arguments or {}, sort_keys=True, default=str)
    return f"{call.name}:{args_str}"


def tool_calls_signature(calls: list[ToolCallContent]) -> str:
    """生成工具调用批次的规范化签名。

    签名生成规则：
    - 排序工具调用：忽略调用顺序，只关注工具集合
    - 分隔符 "|"：避免与 JSON 中的常见字符冲突
    """
    parts = [tool_call_signature(c) for c in calls]
    return "|".join(sorted(parts))


def is_file_mutation_tool(tool_name: str) -> bool:
    """判断工具是否会修改文件。

    文件变更类工具：write_file, edit_file, bash（可能修改文件）
    """
    mutation_tools = {
        "write_file",
        "edit_file",
        "bash",
        "create_file",
        "delete_file",
        "move_file",
        "rename_file",
    }
    return tool_name in mutation_tools


def is_file_read_tool(tool_name: str) -> bool:
    """判断工具是否只读文件。

    只读工具：read_file, grep_search, glob_files, ls
    """
    read_tools = {
        "read_file",
        "grep_search",
        "glob_files",
        "ls",
        "find_files",
    }
    return tool_name in read_tools


def should_clear_read_history(
    new_calls: list[ToolCallContent],
    read_history: list[str],
) -> bool:
    """判断是否应清除只读工具历史。

    规则：如果本批次包含文件变更工具，清除之前的只读调用记录。

    设计原因：避免"编辑后复读"被误判为重复调用。
    """
    return any(is_file_mutation_tool(c.name) for c in new_calls)


def is_tool_productive_default(
    tool_calls: list[ToolCallContent],
    tool_results: list[ToolResultMessage],
) -> bool:
    """默认生产力检查：有任何非错误结果即视为有生产力。"""
    return any(not r.is_error for r in tool_results)


def update_repeated_tool_watchdog(
    state: _LoopRunState,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
) -> str | None:
    """检测工具调用是否重复，防止无限循环。

    比较工具签名而非工具名的原因：
    - 工具名相同但参数不同视为有效重试（如搜索不同关键词）
    - 签名完全相同（包括参数）才视为无效重复
    """
    sig = tool_calls_signature(tool_calls)
    if sig == state.last_tool_signature:
        state.repeated_tool_count += 1
    else:
        state.repeated_tool_count = 0
        state.last_tool_signature = sig

    if (
        config.watchdog_repeated_tool_limit > 0
        and state.repeated_tool_count >= config.watchdog_repeated_tool_limit
    ):
        return f"watchdog stopped repeated tool call: {tool_calls[0].name}"
    return None


def update_idle_tool_watchdog(
    state: _LoopRunState,
    tool_calls: list[ToolCallContent],
    tool_results: list[ToolResultMessage],
    config: AgentLoopConfig,
) -> str | None:
    is_productive = config.is_tool_productive or is_tool_productive_default
    if is_productive(tool_calls, tool_results):
        state.consecutive_idle_steps = 0
    else:
        state.consecutive_idle_steps += 1

    if (
        config.max_consecutive_idle_steps > 0
        and state.consecutive_idle_steps >= config.max_consecutive_idle_steps
    ):
        return (
            f"Watchdog triggered: {state.consecutive_idle_steps} consecutive steps "
            f"without productive tool calls."
        )
    return None
