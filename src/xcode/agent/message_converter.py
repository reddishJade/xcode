"""Agent 消息到 LLM provider 格式的转换逻辑。

从 messages.py 提取，与消息类型定义分离。
"""

from __future__ import annotations

from typing import Any

import orjson

from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    BranchSummaryMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.protocols import ContentBlock
from xcode.agent.types import (
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
)


# 使用 <summary> XML 标签包裹压缩摘要的设计原因：
# 1. 结构化标记：便于 LLM 区分摘要内容与正常对话
# 2. 显式边界：避免摘要文本与后续消息混淆
# 3. 解析友好：工具可通过 XML 标签提取摘要用于分析或审计
COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
SUMMARY_SUFFIX = "\n</summary>"


def convert_to_llm(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    return [_convert_one(m) for m in messages]


def _convert_one(m: AgentMessage) -> dict[str, Any]:
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": str(m.content)}

    if isinstance(m, UserMessage):
        return {"role": "user", "content": m.content}

    if isinstance(m, AssistantMessage):
        return _convert_assistant(m)

    if isinstance(m, ToolResultMessage):
        return _convert_tool_result(m)

    if isinstance(m, BranchSummaryMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": BRANCH_SUMMARY_PREFIX + m.summary + SUMMARY_SUFFIX,
                }
            ],
        }

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": COMPACTION_SUMMARY_PREFIX + m.summary + SUMMARY_SUFFIX,
            }
        ],
    }


def _convert_tool_result(m: ToolResultMessage) -> dict[str, Any]:
    """将工具结果转换为 provider 边界格式。"""
    return {
        "role": "tool",
        "tool_call_id": m.tool_call_id,
        "content": _tool_result_content_text(m.content),
    }


def _tool_result_content_text(content: object) -> str:
    """将工具结果内容压平成 provider 可接受的文本。"""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, TextContent):
                parts.append(item.text)
            elif isinstance(item, ImageContent):
                parts.append(_image_result_summary(item))
            elif isinstance(item, FileContent):
                parts.append(_file_result_summary(item))
            elif isinstance(item, ShellCallOutputContent):
                parts.append(str(item.output))
            elif isinstance(item, ToolResultContent):
                parts.append(item.content)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _image_result_summary(content: ImageContent) -> str:
    """将图片结果压缩为不含二进制数据的 provider 文本。"""
    source = content.source or {}
    media_type = source.get("media_type")
    suffix = f": {media_type}" if isinstance(media_type, str) else ""
    return f"[image result{suffix}]"


def _file_result_summary(content: FileContent) -> str:
    """将文件结果压缩为不含内联数据的 provider 文本。"""
    identity = content.filename or content.file_id or "unnamed"
    return f"[file result: {identity}]"


def _convert_block(block: ContentBlock) -> dict[str, Any] | None:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCallContent):
        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": orjson.dumps(block.arguments or {}).decode(),
            },
        }
    if isinstance(block, ThinkingContent):
        return None
    return None


def _convert_assistant(m: AssistantMessage) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    thinking_parts: list[str] = []
    for block in m.content:
        if isinstance(block, ThinkingContent):
            thinking_parts.append(block.thinking)
            continue
        converted = _convert_block(block)
        if converted is None:
            continue
        if converted.get("type") == "function":
            tool_calls.append(converted)
        else:
            content_blocks.append(converted)

    result: dict[str, Any] = {"role": "assistant"}
    if m.reasoning_content is not None:
        result["reasoning_content"] = m.reasoning_content
    elif thinking_parts:
        result["reasoning_content"] = "".join(thinking_parts)
    if content_blocks:
        # 允许 content 为 None 的设计原因：
        # OpenAI API 允许纯工具调用消息（仅 tool_calls 无 content）。
        # 当消息只包含工具调用而无文本时，content 应为 None 而非空字符串。
        result["content"] = (
            "".join(
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            )
            or None
        )
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result
