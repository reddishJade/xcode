from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xcode.ai.events import StopReason
from xcode.agent.types import ImageContent, TextContent, ThinkingContent, ToolCallContent

from .protocols import ContentBlock

"""Agent 消息类型与消息转换。"""

# ── 消息类型 ──


@dataclass
class SystemMessage:
    role: str = "system"
    content: str = ""
    timestamp: int = 0


@dataclass
class UserMessage:
    role: str = "user"
    content: str | list[TextContent | ImageContent] = ""
    timestamp: int = 0


@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    reasoning_content: str | None = None
    stop_reason: StopReason = "end_turn"
    error_message: str | None = None
    model: str = ""
    provider: str = ""
    timestamp: int = 0
    usage: dict[str, int] | None = None


@dataclass
class ToolResultMessage:
    role: str = "tool_result"
    tool_call_id: str = ""
    tool_name: str = ""
    content: str | list[TextContent | ImageContent] = ""
    is_error: bool = False
    timestamp: int = 0


@dataclass
class CompactionSummaryMessage:
    role: str = "compaction_summary"
    summary: str = ""
    tokens_before: int = 0
    timestamp: int = 0


@dataclass
class BranchSummaryMessage:
    role: str = "branch_summary"
    summary: str = ""
    from_id: str = ""
    timestamp: int = 0


type AgentMessage = (
    SystemMessage
    | UserMessage
    | AssistantMessage
    | ToolResultMessage
    | CompactionSummaryMessage
    | BranchSummaryMessage
)


# ── 消息转换（LLM 格式）──

# 使用 <summary> XML 标签包裹压缩摘要的设计原因：
# 1. 结构化标记：便于 LLM 区分摘要内容与正常对话
# 2. 显式边界：避免摘要文本与后续消息混淆
# 3. 解析友好：工具可通过 XML 标签提取摘要用于分析或审计
COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
SUMMARY_SUFFIX = "\n</summary>"


def convert_to_llm(messages: list[AgentMessage]) -> list[dict[str, Any]]:
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
        status = "ok"
        if m.is_error:
            status = "interrupted" if "interrupted" in str(m.content) else "error"
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                    "status": status,
                }
            ],
        }

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

    if isinstance(m, CompactionSummaryMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": COMPACTION_SUMMARY_PREFIX
                    + m.summary
                    + SUMMARY_SUFFIX,
                }
            ],
        }

    return None


def _convert_block(block: ContentBlock) -> dict[str, Any] | None:
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
        return None
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
