"""Agent 循环配置与上下文类型。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import SkipValidation

from xcode.ai.providers.protocol import StreamProvider
from xcode.ai.types import StreamOptions, ThinkingLevel
from xcode.agent.types import ToolArguments, ToolCallContent

from .messages import AgentMessage, AssistantMessage, ToolResultMessage
from .protocols import (
    AgentTool,
    AgentToolResult,
    CancellationSignal,
    ToolExecutionMode,
    ToolResultContentBlock,
    ToolResultDetails,
)
from xcode.agent.context_assembly import ContextAssembler
from xcode.agent.context_collector import ContextCollectorRegistry

from .hooks import (
    ArchiveWriter,
    BeforeProviderRequestHook,
    CompactHook,
    ContextTransformer,
    IsToolProductiveHook,
    MessageConverter,
    ShouldCompactHook,
)


class _LoopRunState(BaseModel):
    """Agent 循环运行时状态，由 agent_loop.py 持有并更新。"""

    first_turn: bool = True
    pending_messages: list[AgentMessage] = Field(default_factory=list)
    last_tool_signature: str | None = None
    repeated_tool_count: int = 0
    consecutive_idle_steps: int = 0
    consecutive_continuations: int = 0
    step_retries: int = 0
    active_provider: Annotated[StreamProvider | None, SkipValidation] = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# ── 上下文 ──


class AgentContext(BaseModel):
    system_prompt: str = ""
    messages: list[AgentMessage] = Field(default_factory=list)
    tools: list[Annotated[AgentTool, SkipValidation]] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# ── 钩子上下文 ──


class BeforeToolCallContext(BaseModel):
    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: ToolArguments
    context: AgentContext
    model_config = ConfigDict(extra="forbid")


class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str = ""
    args: ToolArguments | None = None
    model_config = ConfigDict(extra="forbid")


class AfterToolCallContext(BaseModel):
    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: ToolArguments
    result: AgentToolResult
    is_error: bool
    context: AgentContext
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class AfterToolCallResult(BaseModel):
    content: list[ToolResultContentBlock] | None = None
    details: ToolResultDetails | None = None
    is_error: bool | None = None
    terminate: bool | None = None
    model_config = ConfigDict(extra="forbid")


class ShouldStopAfterTurnContext(BaseModel):
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]
    model_config = ConfigDict(extra="forbid")


class AgentLoopTurnUpdate(BaseModel):
    context: AgentContext | None = None
    thinking_level: ThinkingLevel = "off"
    model_config = ConfigDict(extra="forbid")


# ── 压缩指令 ──


type CompactPriority = Literal[
    "architecture_decision",
    "modified_file",
    "verification_status",
    "todo",
    "tool_output",
]


class CompactInstructions(BaseModel):
    priorities: list[CompactPriority] = Field(
        default_factory=lambda: [
            "architecture_decision",
            "modified_file",
            "verification_status",
            "todo",
            "tool_output",
        ]
    )
    frozen_identifiers: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


# ── Callable type aliases（引用本模块上下文类型）──


type BeforeToolCallHook = Callable[
    [BeforeToolCallContext, CancellationSignal | None], BeforeToolCallResult | None
]
type AfterToolCallHook = Callable[
    [AfterToolCallContext, CancellationSignal | None], AfterToolCallResult | None
]
type PrepareNextTurnHook = Callable[[], AgentLoopTurnUpdate | None]
type ShouldStopAfterTurnHook = Callable[[ShouldStopAfterTurnContext], bool]


# ── 循环配置 ──


class AgentLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    provider: Annotated[StreamProvider | None, SkipValidation] = None
    tool_execution: ToolExecutionMode = "parallel"
    tool_workers: int = 4

    convert_to_llm: MessageConverter | None = None
    transform_context: ContextTransformer | None = None

    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    prepare_next_turn: PrepareNextTurnHook | None = None
    should_stop_after_turn: ShouldStopAfterTurnHook | None = None

    max_steps: int = 50

    max_step_retries: int = 3
    retry_backoff_base: float = 0.5

    max_tokens_continuation: bool = True
    max_consecutive_continuations: int = 3
    min_continuation_tokens: int = 500

    # 看门狗限制（经验阈值，可根据实际任务调整）
    watchdog_repeated_tool_limit: int = 3  # 连续重复同一工具签名 3 次则终止
    max_consecutive_idle_steps: int = 4  # 连续 4 次工具调用无产出则终止

    should_compact: ShouldCompactHook | None = None
    compact: CompactHook | None = None
    compact_instructions: CompactInstructions | None = None
    archive_writer: ArchiveWriter | None = None

    is_tool_productive: IsToolProductiveHook | None = None
    before_provider_request: BeforeProviderRequestHook | None = None

    context_collectors: ContextCollectorRegistry | None = None
    """上下文收集器注册表。配置后在 context_assembler 之前执行。

    所有注册的 collector 按顺序运行，输出合并为 context_blocks
    传递给 context_assembler。未配置时收集阶段返回空列表。
    """

    context_assembler: Annotated[ContextAssembler | None, SkipValidation] = None
    """结构化上下文组装器。配置后替代/增强 transform_context 的功能。

    未配置时消息流完全不变。配置后每轮 provider 调用前执行，
    按优先级注入 context_blocks，支持 budget 裁剪和过期过滤。
    与 transform_context 兼容：先执行 context_assembler，再执行 transform_context。
    """

    options: StreamOptions | None = None
