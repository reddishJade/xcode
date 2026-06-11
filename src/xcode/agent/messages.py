from __future__ import annotations

from dataclasses import dataclass, field

from xcode.ai.events import StopReason
from xcode.agent.types import (
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ToolResultContent,
)

from .protocols import ContentBlock

"""Agent 消息类型。"""

type UserContent = str | list[TextContent | ImageContent | FileContent]
type ToolResultMessageContent = (
    str
    | list[
        TextContent
        | ImageContent
        | FileContent
        | ToolResultContent
        | ShellCallOutputContent
    ]
)

# ── 消息类型 ──


@dataclass
class SystemMessage:
    role: str = "system"
    content: str = ""
    timestamp: int = 0


@dataclass
class UserMessage:
    role: str = "user"
    content: UserContent = ""
    timestamp: int = 0


@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    reasoning_content: str | None = None
    phase: str | None = None
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
    content: ToolResultMessageContent = ""
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
