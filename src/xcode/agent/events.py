from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .messages import AgentMessage, ToolResultMessage

"""Agent 事件类型。"""


@dataclass
class AgentStartEvent:
    """Agent 循环开始事件。"""

    type: str = "agent_start"


@dataclass
class AgentEndEvent:
    """Agent 循环结束事件，携带最终消息列表。"""

    type: str = "agent_end"
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    """单次 turn 开始事件（调用模型前）。"""

    type: str = "turn_start"


@dataclass
class TurnEndEvent:
    """单次 turn 结束事件（模型返回并执行工具后）。"""

    type: str = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    """模型响应开始事件（首个 token 到达）。"""

    type: str = "message_start"
    message: AgentMessage | None = None


@dataclass
class MessageUpdateEvent:
    """模型响应增量更新事件（流式传输中）。"""

    type: str = "message_update"
    message: AgentMessage | None = None


@dataclass
class MessageEndEvent:
    """模型响应结束事件（流式传输完成）。"""

    type: str = "message_end"
    message: AgentMessage | None = None


@dataclass
class ToolExecutionStartEvent:
    """工具执行开始事件。"""

    type: str = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None


@dataclass
class ToolExecutionUpdateEvent:
    """工具执行增量更新事件（支持进度报告的工具）。"""

    type: str = "tool_execution_update"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None


@dataclass
class ToolExecutionEndEvent:
    """工具执行结束事件，携带结果或错误。"""

    type: str = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: ToolResultMessage | None = None
    is_error: bool = False


@dataclass
class ThinkingUpdateEvent:
    """思考内容增量更新事件（支持 reasoning_content 的模型）。"""

    type: str = "thinking_update"
    reasoning_content: str = ""


@dataclass
class CompactionArchive:
    """压缩归档元数据。"""

    path: str
    status: Literal["summary", "full"]


@dataclass
class CompactionEvent:
    """上下文压缩事件，包含压缩统计和归档路径。"""

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
