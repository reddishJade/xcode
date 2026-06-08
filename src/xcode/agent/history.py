"""消息历史修复和清理工具。

提供工具调用配对修复、请求 hygiene 和历史压缩辅助功能。
"""

from __future__ import annotations

import re
from typing import Any

from xcode.agent.messages import AgentMessage, AssistantMessage, ToolResultMessage
from xcode.agent.types import ToolCallContent, ToolResultContent


def repair_tool_pairing(messages: list[AgentMessage]) -> list[AgentMessage]:
    """修复工具调用和结果的配对关系。

    规则：
    1. 移除孤儿 tool_result（没有对应 tool_call）
    2. 移除未完成的 tool_call（没有对应 result）
    3. 保持消息顺序和其他内容不变

    设计原因：避免畸形工具历史污染模型上下文和缓存前缀。
    """
    if not messages:
        return messages

    # 收集所有 tool_call id
    tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCallContent):
                    tool_call_ids.add(block.id)

    # 收集所有 tool_result 对应的 tool_use_id
    tool_result_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    tool_result_ids.add(block.tool_use_id)

    # 过滤消息
    repaired: list[AgentMessage] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            # 过滤掉没有 result 的 tool_call
            filtered_content = []
            for block in msg.content:
                if isinstance(block, ToolCallContent):
                    if block.id in tool_result_ids:
                        filtered_content.append(block)
                else:
                    filtered_content.append(block)
            # 如果还有内容，保留消息
            if filtered_content:
                repaired.append(
                    AssistantMessage(
                        content=filtered_content,
                        stop_reason=msg.stop_reason,
                        model=msg.model,
                        usage=msg.usage,
                    )
                )
        elif isinstance(msg, ToolResultMessage):
            # 过滤掉孤儿 tool_result
            filtered_content = []
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    if block.tool_use_id in tool_call_ids:
                        filtered_content.append(block)
                else:
                    filtered_content.append(block)
            # 如果还有内容，保留消息
            if filtered_content:
                repaired.append(
                    ToolResultMessage(content=filtered_content, is_error=msg.is_error)
                )
        else:
            # 其他消息类型原样保留
            repaired.append(msg)

    return repaired


def apply_request_hygiene(
    messages: list[AgentMessage],
    *,
    max_tool_result_bytes: int = 8000,
    max_tool_arg_length: int = 1000,
    keep_head_lines: int = 50,
    keep_tail_lines: int = 50,
) -> list[AgentMessage]:
    """对请求消息历史应用 hygiene 规则。

    规则：
    1. 超大 tool_result 按字节/行数上限保留 head + tail + signal lines
    2. base64 payload 替换为占位符
    3. 已完成工具调用的超长字符串参数替换为占位符

    重要：只在发给模型的请求边界压缩，磁盘/session 保留完整历史。

    设计原因：避免超长工具输出和参数污染缓存热前缀占比，同时保留错误信息。
    """
    cleaned: list[AgentMessage] = []

    # 收集已完成的 tool_call ids
    completed_tool_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    completed_tool_ids.add(block.tool_use_id)

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            # 清理已完成工具调用的超长参数
            cleaned_content = []
            for block in msg.content:
                if isinstance(block, ToolCallContent) and block.id in completed_tool_ids:
                    # 压缩超长参数
                    cleaned_args = _truncate_tool_args(
                        block.arguments or {}, max_tool_arg_length
                    )
                    cleaned_content.append(
                        ToolCallContent(
                            id=block.id,
                            name=block.name,
                            arguments=cleaned_args,
                        )
                    )
                else:
                    cleaned_content.append(block)
            cleaned.append(
                AssistantMessage(
                    content=cleaned_content,
                    stop_reason=msg.stop_reason,
                    model=msg.model,
                    usage=msg.usage,
                )
            )
        elif isinstance(msg, ToolResultMessage):
            # 清理超大 tool_result
            cleaned_content = []
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    cleaned_text = _truncate_tool_result(
                        block.content,
                        max_tool_result_bytes,
                        keep_head_lines,
                        keep_tail_lines,
                    )
                    cleaned_content.append(
                        ToolResultContent(
                            tool_use_id=block.tool_use_id,
                            content=cleaned_text,
                            status=block.status,
                        )
                    )
                else:
                    cleaned_content.append(block)
            cleaned.append(
                ToolResultMessage(content=cleaned_content, is_error=msg.is_error)
            )
        else:
            cleaned.append(msg)

    return cleaned


def _truncate_tool_args(args: dict[str, Any], max_length: int) -> dict[str, Any]:
    """压缩工具参数中的超长字符串。"""
    cleaned = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_length:
            cleaned[key] = f"<truncated, {len(value)} chars>"
        elif isinstance(value, dict):
            cleaned[key] = _truncate_tool_args(value, max_length)
        elif isinstance(value, list):
            cleaned[key] = [
                _truncate_tool_args(item, max_length) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _truncate_tool_result(
    content: str,
    max_bytes: int,
    keep_head_lines: int,
    keep_tail_lines: int,
) -> str:
    """压缩超大工具结果，保留 head + tail + signal lines。"""
    # 检查是否包含 base64 payload
    if _is_base64_payload(content):
        return f"<base64 data, {len(content)} bytes>"

    # 按行处理
    lines = content.splitlines()

    # 检查行数是否超过阈值
    if len(lines) <= keep_head_lines + keep_tail_lines:
        # 行数未超标，检查字节大小
        content_bytes = content.encode("utf-8", errors="ignore")
        if len(content_bytes) <= max_bytes:
            return content

    # 需要压缩
    if len(lines) <= keep_head_lines + keep_tail_lines:
        # 行数少但字节多，直接截断
        return content[:max_bytes] + f"\n... (truncated, {len(content)} bytes total) ..."

    # 提取 signal lines（错误/警告）
    signal_lines = []
    for i, line in enumerate(lines):
        if _is_signal_line(line):
            signal_lines.append((i, line))

    # 构建压缩结果
    head = lines[:keep_head_lines]
    tail = lines[-keep_tail_lines:]

    # 添加 signal lines（避免重复）
    middle_signals = [
        line for i, line in signal_lines
        if i >= keep_head_lines and i < len(lines) - keep_tail_lines
    ]

    parts = head
    if middle_signals:
        parts.append(f"\n... ({len(lines) - keep_head_lines - keep_tail_lines} lines omitted) ...\n")
        parts.extend(middle_signals)
    else:
        parts.append(f"\n... ({len(lines) - keep_head_lines - keep_tail_lines} lines omitted) ...\n")
    parts.extend(tail)

    return "\n".join(parts)


def _is_base64_payload(content: str) -> bool:
    """检测是否为 base64 payload。"""
    if len(content) < 100:
        return False
    # 简单检测：连续 base64 字符比例 > 90%
    base64_chars = re.findall(r"[A-Za-z0-9+/=]", content)
    return len(base64_chars) / len(content) > 0.9


def _is_signal_line(line: str) -> bool:
    """检测是否为重要信号行（错误/警告）。"""
    line_lower = line.lower()
    keywords = ["error", "exception", "warning", "failed", "traceback", "assert"]
    return any(keyword in line_lower for keyword in keywords)
