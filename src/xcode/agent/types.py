from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal, Protocol

"""Agent 核心类型。

基于 TS pi/packages/agent/src/types.ts 设计，定义了 Agent 循环所需的
所有类型：消息、事件、工具定义、配置和队列模式。
"""

# ── 基础 ──

type QueueMode = Literal["all", "one-at-a-time"]
type ToolExecutionMode = Literal["sequential", "parallel"]
type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
type StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "error", "aborted"]

# ── 内容块 ──


@dataclass(frozen=True)
class TextContent:
    type: str = "text"
    text: str = ""


@dataclass(frozen=True)
class ImageContent:
    type: str = "image"
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCallBlock:
    type: str = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThinkingBlock:
    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None


type ContentBlock = TextContent | ImageContent | ToolCallBlock | ThinkingBlock

# ── 消息 ──


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
    role: str = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: str | list[TextContent | ImageContent] = ""
    is_error: bool = False
    timestamp: int = 0


@dataclass
class CustomMessage:
    role: str = "custom"
    custom_type: str = ""
    content: str | list[TextContent | ImageContent] = ""
    display: bool = True
    details: Any = None
    timestamp: int = 0


@dataclass
class BashExecutionMessage:
    role: str = "bashExecution"
    command: str = ""
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    timestamp: int = 0


@dataclass
class CompactionSummaryMessage:
    role: str = "compactionSummary"
    summary: str = ""
    tokens_before: int = 0
    timestamp: int = 0


@dataclass
class BranchSummaryMessage:
    role: str = "branchSummary"
    summary: str = ""
    from_id: str = ""
    timestamp: int = 0


type AgentMessage = (
    SystemMessage
    | UserMessage
    | AssistantMessage
    | ToolResultMessage
    | CustomMessage
    | BashExecutionMessage
    | CompactionSummaryMessage
    | BranchSummaryMessage
)

# ── 工具定义 ──


@dataclass
class ToolDefinition:
    """工具 schema 定义：LLM 可见的部分（名称、描述、参数 schema）。"""

    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema（与 ToolSpec 保持一致的字段名）
    execution_mode: ToolExecutionMode | None = None


@dataclass
class AgentToolResult[T]:
    content: list[TextContent | ImageContent]
    details: T | None = None
    terminate: bool = False


type ToolUpdateCallback = Callable[[AgentToolResult[Any]], None]


class AgentTool[Details](Protocol):
    """工具运行时：在 ToolDefinition 基础上增加执行上下文。"""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execution_mode: ToolExecutionMode | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult[Details]: ...


def tool_definition_from_spec(spec: Any) -> ToolDefinition:
    """从 ToolSpec（或其他兼容对象）提取 ToolDefinition。"""
    return ToolDefinition(
        name=getattr(spec, "name", ""),
        description=getattr(spec, "description", ""),
        schema=getattr(spec, "schema", {}) or {},
        execution_mode=getattr(spec, "execution_mode", None),
    )


# ── 上下文 ──


@dataclass
class AgentContext:
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool[Any]] = field(default_factory=list)


# ── 事件 ──


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
    result: Any = None
    is_error: bool = False


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
)

# ── 监听器 ──

type AgentListener = Callable[
    [AgentEvent, Any], None
]  # signal is Any (AbortSignal | CancellationToken)

# ── 循环配置 ──


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallBlock
    args: dict[str, Any]
    context: AgentContext


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str = ""


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallBlock
    args: dict[str, Any]
    result: AgentToolResult[Any]
    is_error: bool
    context: AgentContext


@dataclass
class AfterToolCallResult:
    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]


@dataclass
class AgentLoopTurnUpdate:
    context: AgentContext | None = None
    model: Any = None
    thinking_level: ThinkingLevel = "off"


@dataclass
class AgentEventSink(Protocol):
    async def __call__(self, event: AgentEvent) -> None: ...


type StreamFn = Callable[..., AsyncIterator[Any]]


@dataclass
class AgentLoopConfig:
    model: Any  # Model object
    reasoning: str | None = None
    session_id: str | None = None
    transport: str = "auto"
    thinking_budgets: Any = None
    max_retry_delay_ms: int | None = None
    tool_execution: ToolExecutionMode = "parallel"
    api_key: str | None = None

    # 转换函数
    convert_to_llm: Callable[[list[AgentMessage]], list[dict[str, Any]]] | None = None
    transform_context: (
        Callable[[list[AgentMessage], Any], list[AgentMessage]] | None
    ) = None
    get_api_key: Callable[[str], str | None] | None = None

    # 钩子
    before_tool_call: (
        Callable[[BeforeToolCallContext, Any], BeforeToolCallResult | None] | None
    ) = None
    after_tool_call: (
        Callable[[AfterToolCallContext, Any], AfterToolCallResult | None] | None
    ) = None
    prepare_next_turn: Callable[[], AgentLoopTurnUpdate | None] | None = None
    should_stop_after_turn: Callable[[ShouldStopAfterTurnContext], bool] | None = None

    # 队列
    get_steering_messages: Callable[[], list[AgentMessage]] | None = None
    get_follow_up_messages: Callable[[], list[AgentMessage]] | None = None


# ── Agent State ──


class AgentState(Protocol):
    system_prompt: str
    model: Any
    thinking_level: ThinkingLevel
    messages: list[AgentMessage]
    tools: list[AgentTool[Any]]
    is_streaming: bool
    streaming_message: AgentMessage | None
    pending_tool_calls: set[str]
    error_message: str | None
