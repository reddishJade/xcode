"""消息历史修复和请求卫生。"""

from __future__ import annotations

import re

from xcode.agent.messages import AgentMessage, AssistantMessage, ToolResultMessage
from xcode.agent.types import ToolArguments, ToolCallContent


def repair_tool_pairing(messages: list[AgentMessage]) -> list[AgentMessage]:
    if not messages:
        return messages

    tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCallContent):
                    tool_call_ids.add(block.id)

    tool_result_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolResultMessage) and msg.tool_call_id:
            tool_result_ids.add(msg.tool_call_id)

    repaired: list[AgentMessage] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            filtered_content = []
            for block in msg.content:
                if isinstance(block, ToolCallContent):
                    if block.id in tool_result_ids:
                        filtered_content.append(block)
                else:
                    filtered_content.append(block)
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
            if msg.tool_call_id in tool_call_ids:
                repaired.append(msg)
        else:
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
    cleaned: list[AgentMessage] = []

    completed_tool_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolResultMessage) and msg.tool_call_id:
            completed_tool_ids.add(msg.tool_call_id)

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            cleaned_content = []
            for block in msg.content:
                if (
                    isinstance(block, ToolCallContent)
                    and block.id in completed_tool_ids
                ):
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
            if isinstance(msg.content, str):
                truncated = _truncate_tool_result(
                    msg.content, max_tool_result_bytes, keep_head_lines, keep_tail_lines
                )
                cleaned.append(
                    ToolResultMessage(
                        tool_call_id=msg.tool_call_id,
                        tool_name=msg.tool_name,
                        content=truncated,
                        is_error=msg.is_error,
                    )
                )
            else:
                from xcode.agent.types import TextContent

                cleaned_blocks = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        cleaned_blocks.append(
                            TextContent(
                                text=_truncate_tool_result(
                                    block.text,
                                    max_tool_result_bytes,
                                    keep_head_lines,
                                    keep_tail_lines,
                                )
                            )
                        )
                    else:
                        cleaned_blocks.append(block)
                cleaned.append(
                    ToolResultMessage(
                        tool_call_id=msg.tool_call_id,
                        tool_name=msg.tool_name,
                        content=cleaned_blocks,
                        is_error=msg.is_error,
                    )
                )
        else:
            cleaned.append(msg)

    return cleaned


def _truncate_tool_args(args: ToolArguments, max_length: int) -> ToolArguments:
    cleaned: ToolArguments = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_length:
            cleaned[key] = f"<truncated, {len(value)} chars>"
        elif isinstance(value, dict):
            cleaned[key] = _truncate_tool_args(value, max_length)
        elif isinstance(value, list):
            cleaned[key] = [
                _truncate_tool_args(item, max_length)
                if isinstance(item, dict)
                else item
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
    if _is_base64_payload(content):
        return f"<base64 data, {len(content)} bytes>"

    lines = content.splitlines()

    if len(lines) <= keep_head_lines + keep_tail_lines:
        content_bytes = content.encode("utf-8", errors="ignore")
        if len(content_bytes) <= max_bytes:
            return content

    if len(lines) <= keep_head_lines + keep_tail_lines:
        return (
            content[:max_bytes] + f"\n... (truncated, {len(content)} bytes total) ..."
        )

    signal_lines = []
    for i, line in enumerate(lines):
        if _is_signal_line(line):
            signal_lines.append((i, line))

    head = lines[:keep_head_lines]
    tail = lines[-keep_tail_lines:]

    middle_signals = [
        line
        for i, line in signal_lines
        if i >= keep_head_lines and i < len(lines) - keep_tail_lines
    ]

    parts = head
    if middle_signals:
        parts.append(
            f"\n... ({len(lines) - keep_head_lines - keep_tail_lines} lines omitted) ...\n"
        )
        parts.extend(middle_signals)
    else:
        parts.append(
            f"\n... ({len(lines) - keep_head_lines - keep_tail_lines} lines omitted) ...\n"
        )
    parts.extend(tail)

    return "\n".join(parts)


def _is_base64_payload(content: str) -> bool:
    if len(content) < 100:
        return False
    base64_chars = re.findall(r"[A-Za-z0-9+/=]", content)
    return len(base64_chars) / len(content) > 0.9


def _is_signal_line(line: str) -> bool:
    line_lower = line.lower()
    keywords = ["error", "exception", "warning", "failed", "traceback", "assert"]
    return any(keyword in line_lower for keyword in keywords)
