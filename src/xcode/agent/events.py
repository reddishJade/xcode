from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .messages import AgentMessage, ToolResultMessage
from .protocols import AgentToolResult
from .types import ToolArguments

"""Agent 事件类型。"""


class AgentStartEvent(BaseModel):
    """Agent 循环开始事件。"""

    type: str = "agent_start"
    model_config = ConfigDict(extra="forbid")


class AgentEndEvent(BaseModel):
    """Agent 循环结束事件，携带最终消息列表。"""

    type: str = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class TurnStartEvent(BaseModel):
    """单次 turn 开始事件（调用模型前）。"""

    type: str = "turn_start"
    model_config = ConfigDict(extra="forbid")


class TurnEndEvent(BaseModel):
    """单次 turn 结束事件（模型返回并执行工具后）。"""

    type: str = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class MessageStartEvent(BaseModel):
    """模型响应开始事件（首个 token 到达）。"""

    type: str = "message_start"
    message: AgentMessage | None = None
    model_config = ConfigDict(extra="forbid")


class MessageUpdateEvent(BaseModel):
    """模型响应增量更新事件（流式传输中）。"""

    type: str = "message_update"
    message: AgentMessage | None = None
    model_config = ConfigDict(extra="forbid")


class MessageEndEvent(BaseModel):
    """模型响应结束事件（流式传输完成）。"""

    type: str = "message_end"
    message: AgentMessage | None = None
    model_config = ConfigDict(extra="forbid")


class ToolExecutionStartEvent(BaseModel):
    """工具执行开始事件。"""

    type: str = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: ToolArguments = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ToolExecutionUpdateEvent(BaseModel):
    """工具执行增量更新事件（支持进度报告的工具）。"""

    type: str = "tool_execution_update"
    tool_call_id: str = ""
    tool_name: str = ""
    args: ToolArguments = Field(default_factory=dict)
    partial_result: AgentToolResult | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ToolExecutionEndEvent(BaseModel):
    """工具执行结束事件，携带结果或错误。"""

    type: str = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: ToolResultMessage | None = None
    is_error: bool = False
    model_config = ConfigDict(extra="forbid")


class ThinkingUpdateEvent(BaseModel):
    """思考内容增量更新事件（支持 reasoning_content 的模型）。"""

    type: str = "thinking_update"
    reasoning_content: str = ""
    model_config = ConfigDict(extra="forbid")


class CompactionArchive(BaseModel):
    """压缩归档元数据。"""

    path: str
    status: Literal["summary", "full"]
    model_config = ConfigDict(extra="forbid")


class CompactionEvent(BaseModel):
    """上下文压缩事件，包含压缩统计和归档路径。"""

    type: str = "compaction"
    messages_removed: int = 0
    messages_after: int = 0
    summary_token_estimate: int = 0
    trigger: str = "token_limit"
    archive: CompactionArchive | None = None
    model_config = ConfigDict(extra="forbid")


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
