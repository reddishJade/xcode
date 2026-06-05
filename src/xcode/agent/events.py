from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .messages import AgentMessage, ToolResultMessage

"""Agent 事件类型。"""


@dataclass
class AgentStartEvent:
    type: str = "agent_start"


@dataclass
class AgentEndEvent:
    type: str = "agent_end"
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    type: str = "turn_start"


@dataclass
class TurnEndEvent:
    type: str = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    type: str = "message_start"
    message: AgentMessage | None = None


@dataclass
class MessageUpdateEvent:
    type: str = "message_update"
    message: AgentMessage | None = None


@dataclass
class MessageEndEvent:
    type: str = "message_end"
    message: AgentMessage | None = None


@dataclass
class ToolExecutionStartEvent:
    type: str = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None


@dataclass
class ToolExecutionUpdateEvent:
    type: str = "tool_execution_update"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None


@dataclass
class ToolExecutionEndEvent:
    type: str = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: ToolResultMessage | None = None
    is_error: bool = False


@dataclass
class ThinkingUpdateEvent:
    type: str = "thinking_update"
    reasoning_content: str = ""


@dataclass
class CompactionArchive:
    path: str
    status: Literal["summary", "full"]


@dataclass
class CompactionEvent:
    type: str = "compaction"
    messages_removed: int = 0
    messages_after: int = 0
    summary_token_estimate: int = 0
    trigger: str = "token_limit"
    archive: CompactionArchive | None = None


type AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ThinkingUpdateEvent
    | CompactionEvent
)
