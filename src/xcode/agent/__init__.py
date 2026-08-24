"""Xcode Agent — 类型化 Agent 循环。"""

from __future__ import annotations

from .agent import Agent
from .agent_loop import run_agent_loop
from .config import AgentContext, AgentLoopConfig
from .context import (
    ActiveDiffCollector,
    ContextAssembler,
    ContextAssemblyInput,
    ContextAssemblyResult,
    ContextBlock,
    ContextBlockSource,
    ContextCollectionInput,
    ContextCollector,
    ContextCollectorRegistry,
    ContextExpiry,
    ContextPriority,
    DefaultContextAssembler,
    FrozenContextCollectorRegistry,
    InstructionCollector,
    InstructionSource,
    NotesCollector,
    RecentValidationCollector,
    trim_to_budget,
)
from .events import AgentEvent
from .messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from .request import (
    DefaultRequestAssembler,
    RequestAssembler,
    RequestAssembly,
    RequestContextTrace,
    RequestHygiene,
)
from .results import AgentLoopMetrics, AgentLoopResult, TerminationReason
from .types import AgentTool, CancellationSignal

__all__ = [
    "ActiveDiffCollector",
    "Agent",
    "AgentContext",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentLoopMetrics",
    "AgentLoopResult",
    "AgentMessage",
    "AgentTool",
    "AssistantMessage",
    "CancellationSignal",
    "ContextAssembler",
    "ContextAssemblyInput",
    "ContextAssemblyResult",
    "ContextBlock",
    "ContextBlockSource",
    "ContextCollectionInput",
    "ContextCollector",
    "ContextCollectorRegistry",
    "ContextExpiry",
    "ContextPriority",
    "DefaultContextAssembler",
    "DefaultRequestAssembler",
    "FrozenContextCollectorRegistry",
    "InstructionCollector",
    "InstructionSource",
    "NotesCollector",
    "RecentValidationCollector",
    "RequestAssembler",
    "RequestAssembly",
    "RequestContextTrace",
    "RequestHygiene",
    "SystemMessage",
    "TerminationReason",
    "ToolResultMessage",
    "UserMessage",
    "run_agent_loop",
    "trim_to_budget",
]
