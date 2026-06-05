from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import StreamOptions, ThinkingLevel, ToolCallContent

from .messages import AgentMessage, AssistantMessage, ToolResultMessage
from .protocols import (
    AgentTool,
    AgentToolResult,
    CancellationSignal,
    ImageContent,
    QueueMode,
    TextContent,
    ToolExecutionMode,
)

"""Agent 循环配置与上下文类型。"""

# ── 上下文 ──


@dataclass
class AgentContext:
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)


# ── 钩子上下文 ──


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
    result: AgentToolResult
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


# ── 压缩指令 ──


type CompactPriority = Literal[
    "architecture_decision",
    "modified_file",
    "verification_status",
    "todo",
    "tool_output",
]


@dataclass
class CompactInstructions:
    priorities: list[CompactPriority] = field(
        default_factory=lambda: [
            "architecture_decision",
            "modified_file",
            "verification_status",
            "todo",
            "tool_output",
        ]
    )
    frozen_identifiers: list[str] = field(default_factory=list)


# ── Callable type aliases ──


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
type ArchiveWriter = Callable[[list[AgentMessage]], str | None]
type ShouldCompactHook = Callable[[list[AgentMessage]], bool]
type CompactHook = Callable[[list[AgentMessage]], list[AgentMessage]]
type IsToolProductiveHook = Callable[
    [list[ToolCallContent], list[ToolResultMessage]], bool
]


# ── 指标 ──


@dataclass
class AgentLoopMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    model_latencies_ms: list[float] = field(default_factory=list)
    tool_latencies_ms: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# ── 结果 ──


@dataclass
class AgentLoopResult:
    messages: list[AgentMessage] = field(default_factory=list)
    steps: int = 0
    stopped_by_limit: bool = False
    stopped_by_watchdog: bool = False
    stopped_by_error: bool = False
    watchdog_reason: str | None = None
    metrics: AgentLoopMetrics | None = None
    active_provider: ModelProvider | None = None


# ── 循环配置 ──


@dataclass
class AgentLoopConfig:
    provider: ModelProvider | None = None
    tool_execution: ToolExecutionMode = "parallel"

    convert_to_llm: MessageConverter | None = None
    transform_context: ContextTransformer | None = None

    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    prepare_next_turn: PrepareNextTurnHook | None = None
    should_stop_after_turn: ShouldStopAfterTurnHook | None = None

    get_steering_messages: MessageQueueGetter | None = None
    get_follow_up_messages: MessageQueueGetter | None = None
    steering_mode: QueueMode = "all"
    follow_up_mode: QueueMode = "all"

    max_steps: int = 50

    max_step_retries: int = 3
    retry_backoff_base: float = 0.5

    max_tokens_continuation: bool = True
    max_consecutive_continuations: int = 3
    min_continuation_tokens: int = 500

    watchdog_repeated_tool_limit: int = 3
    max_consecutive_idle_steps: int = 4

    should_compact: ShouldCompactHook | None = None
    compact: CompactHook | None = None
    compact_instructions: CompactInstructions | None = None
    archive_writer: ArchiveWriter | None = None

    is_tool_productive: IsToolProductiveHook | None = None

    options: StreamOptions | None = None
