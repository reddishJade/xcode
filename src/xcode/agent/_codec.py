"""Agent 消息到 LLM provider 格式的转换。"""

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
from xcode.agent.types import (
    ContentBlock,
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
)

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
    return {
        "role": "tool",
        "tool_call_id": m.tool_call_id,
        "content": _tool_result_content_text(m.content),
    }


def _tool_result_content_text(content: object) -> str:
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
    source = content.source or {}
    media_type = source.get("media_type")
    suffix = f": {media_type}" if isinstance(media_type, str) else ""
    return f"[image result{suffix}]"


def _file_result_summary(content: FileContent) -> str:
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
        result["content"] = (
            "".join(
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            )
            or None
        )
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result
