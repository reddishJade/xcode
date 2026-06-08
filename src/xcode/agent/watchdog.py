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


class RepeatDetector:
    """重复工具调用检测器（带文件变更感知）。

    用法：
        detector = RepeatDetector(limit=3)
        if detector.check_and_update(tool_calls):
            # 触发重复限制
            pass
    """

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.last_signature: str | None = None
        self.repeat_count: int = 0
        self.read_history: list[str] = []

    def check_and_update(self, calls: list[ToolCallContent]) -> tuple[bool, str | None]:
        """检查并更新重复计数。

        返回：(is_repeated, reason)
        - is_repeated: 是否触发重复限制
        - reason: 触发原因（如果触发）
        """
        if not calls:
            return False, None

        # 检查是否有文件变更工具
        if should_clear_read_history(calls, self.read_history):
            # 清除只读历史
            self.read_history.clear()

        # 生成签名
        signature = tool_calls_signature(calls)

        # 检查是否与上次相同
        if signature == self.last_signature:
            self.repeat_count += 1
            if self.repeat_count >= self.limit:
                reason = (
                    f"工具调用连续重复 {self.repeat_count} 次，"
                    f"签名：{signature[:100]}..."
                )
                return True, reason
        else:
            # 不同签名，重置计数
            self.repeat_count = 1
            self.last_signature = signature

            # 记录只读调用
            for call in calls:
                if is_file_read_tool(call.name):
                    self.read_history.append(tool_call_signature(call))

        return False, None

    def reset(self) -> None:
        """重置检测器状态。"""
        self.last_signature = None
        self.repeat_count = 0
        self.read_history.clear()
