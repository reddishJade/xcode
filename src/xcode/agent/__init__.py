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
    InstructionCollector,
    InstructionSource,
    NotesCollector,
    RecentValidationCollector,
    TaskStateCollector,
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
from .types import AgentTool, CancellationSignal
from .results import AgentLoopMetrics, AgentLoopResult, TerminationReason

__all__ = [
    "ActiveDiffCollector",
    "Agent",
    "AgentContext",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentLoopMetrics",
    "AgentLoopResult",
    "TerminationReason",
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
    "InstructionCollector",
    "InstructionSource",
    "NotesCollector",
    "RecentValidationCollector",
    "run_agent_loop",
    "SystemMessage",
    "TaskStateCollector",
    "ToolResultMessage",
    "trim_to_budget",
    "UserMessage",
]
