from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import (
    ImageContent,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCallContent,
)

"""Agent 核心类型。

定义 Xcode Agent 循环所需的消息、事件、工具定义、配置和队列模式。
"""

# ── 基础 ──

type QueueMode = Literal["all", "one-at-a-time"]
type ToolExecutionMode = Literal["sequential", "parallel"]
type StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "error", "aborted"]

# ── 内容块（来自 ai/types.py）──

type ContentBlock = TextContent | ImageContent | ToolCallContent | ThinkingContent

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
    role: str = "tool_result"
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
    role: str = "bash_execution"
    command: str = ""
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
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
    | CustomMessage
    | BashExecutionMessage
    | CompactionSummaryMessage
    | BranchSummaryMessage
)


@dataclass
class AgentToolResult[T]:
    content: list[TextContent | ImageContent]
    details: T | None = None
    terminate: bool = False


type ToolUpdateCallback = Callable[[AgentToolResult[Any]], None]


class CancellationSignal(Protocol):
    """Agent core 可见的取消信号。"""

    reason: str

    def is_cancelled(self) -> bool: ...


class AgentTool[Details](Protocol):
    """Agent core 可调用的工具运行时接口。"""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execution_mode: ToolExecutionMode | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult[Details]: ...


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

type AgentListener = Callable[[AgentEvent, Any], None]

# ── 循环配置 ──


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, Any]
    context: AgentContext


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str = ""


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallContent
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


type MessageConverter = Callable[[list[AgentMessage]], list[dict[str, Any]]]
type ContextTransformer = Callable[
    [list[AgentMessage], CancellationSignal | None], list[AgentMessage]
]
type BeforeToolCallHook = Callable[
    [BeforeToolCallContext, CancellationSignal | None], BeforeToolCallResult | None
]
type AfterToolCallHook = Callable[
    [AfterToolCallContext, CancellationSignal | None], AfterToolCallResult | None
]
type PrepareNextTurnHook = Callable[[], AgentLoopTurnUpdate | None]
type ShouldStopAfterTurnHook = Callable[[ShouldStopAfterTurnContext], bool]
type MessageQueueGetter = Callable[[], list[AgentMessage]]
type ShouldCompactHook = Callable[[list[AgentMessage]], bool]
type CompactHook = Callable[[list[AgentMessage]], list[AgentMessage]]
type IsToolProductiveHook = Callable[
    [list[ToolCallContent], list[ToolResultMessage]], bool
]


@dataclass
class AgentLoopMetrics:
    """Agent 循环运行指标。"""

    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    model_latencies_ms: list[float] = field(default_factory=list)
    tool_latencies_ms: list[float] = field(default_factory=list)


@dataclass
class AgentLoopResult:
    """Agent 循环执行结果。"""

    messages: list[AgentMessage] = field(default_factory=list)
    steps: int = 0
    stopped_by_limit: bool = False
    stopped_by_watchdog: bool = False
    watchdog_reason: str | None = None
    metrics: AgentLoopMetrics | None = None
    active_provider: Any = None


@dataclass
class AgentLoopConfig:
    provider: ModelProvider | None = None
    tool_execution: ToolExecutionMode = "parallel"

    # 转换函数
    convert_to_llm: MessageConverter | None = None
    transform_context: ContextTransformer | None = None

    # 钩子
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    prepare_next_turn: PrepareNextTurnHook | None = None
    should_stop_after_turn: ShouldStopAfterTurnHook | None = None

    # 队列
    get_steering_messages: MessageQueueGetter | None = None
    get_follow_up_messages: MessageQueueGetter | None = None

    # 步骤控制
    max_steps: int = 50

    # 错误重试
    max_step_retries: int = 3
    retry_backoff_base: float = 0.5

    # max_tokens 续写
    max_tokens_continuation: bool = True
    max_consecutive_continuations: int = 3
    min_continuation_tokens: int = 500

    # 看门狗
    watchdog_repeated_tool_limit: int = 3
    max_consecutive_idle_steps: int = 4

    # Fallback provider
    fallback_provider: ModelProvider | None = None
    consecutive_error_threshold: int = 3

    # 压缩钩子
    should_compact: ShouldCompactHook | None = None
    compact: CompactHook | None = None

    # 生产力检查（空闲步骤看门狗）
    is_tool_productive: IsToolProductiveHook | None = None


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
