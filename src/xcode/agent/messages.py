from __future__ import annotations

from typing import Any

from .types import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    ContentBlock,
    CustomMessage,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)

"""消息转换层。将内部消息角色转为 provider 可接收的消息格式。"""

COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
BRANCH_SUMMARY_SUFFIX = "\n</summary>"


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    """将 BashExecutionMessage 转为纯文本描述。"""
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated:
        text += "\n\n[Output truncated]"
    return text


def convert_to_llm(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """将 AgentMessage list 转换成 LLM 兼容的 dict message list。"""
    result: list[dict[str, Any]] = []
    for m in messages:
        converted = _convert_one(m)
        if converted is not None:
            result.append(converted)
    return result


def _convert_one(m: AgentMessage) -> dict[str, Any] | None:
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": str(m.content)}

    if isinstance(m, UserMessage):
        return {"role": "user", "content": m.content}

    if isinstance(m, AssistantMessage):
        return _convert_assistant(m)

    if isinstance(m, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content,
        }

    if isinstance(m, BashExecutionMessage):
        return {
            "role": "user",
            "content": [{"type": "text", "text": bash_execution_to_text(m)}],
        }

    if isinstance(m, CustomMessage):
        content: Any = m.content
        if isinstance(m.content, str):
            content = [{"type": "text", "text": m.content}]
        return {"role": "user", "content": content}

    if isinstance(m, BranchSummaryMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": BRANCH_SUMMARY_PREFIX + m.summary + BRANCH_SUMMARY_SUFFIX,
                }
            ],
        }

    if isinstance(m, CompactionSummaryMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": COMPACTION_SUMMARY_PREFIX
                    + m.summary
                    + COMPACTION_SUMMARY_SUFFIX,
                }
            ],
        }

    return None


def _convert_block(block: ContentBlock) -> dict[str, Any] | None:
    """将单个 ContentBlock 转为 dict（text/tool_call/thinking）。"""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCallContent):
        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": block.arguments or {},
            },
        }
    if isinstance(block, ThinkingContent):
        return {"type": "text", "text": block.thinking}
    # ImageContent 在 LLM 输出中不直接转换为 text
    return None


def _convert_assistant(m: AssistantMessage) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block in m.content:
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
