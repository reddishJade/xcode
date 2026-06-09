"""重复工具调用检测和抑制增强。

提供带文件变更感知的工具调用签名生成和重复检测。
"""

from __future__ import annotations

import json

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



