"""Xcode Agent — 类型化 Agent 循环。"""

from __future__ import annotations

from .agent import Agent
from .agent_loop import run_agent_loop
from .config import AgentContext, AgentLoopConfig
from .context_assembly import (
    ContextAssembler,
    ContextAssemblyInput,
    ContextAssemblyResult,
    ContextBlock,
    ContextBlockSource,
    ContextExpiry,
    ContextPriority,
    DefaultContextAssembler,
    trim_to_budget,
)
from .context_collector import (
    ActiveDiffCollector,
    ContextCollectionInput,
    ContextCollector,
    ContextCollectorRegistry,
    NotesCollector,
    ProjectManifestCollector,
    RecentValidationCollector,
    TaskStateCollector,
)
from .events import AgentEvent
from .messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from .protocols import AgentTool, CancellationSignal
from .results import AgentLoopMetrics, AgentLoopResult

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
    "NotesCollector",
    "ProjectManifestCollector",
    "RecentValidationCollector",
    "run_agent_loop",
    "SystemMessage",
    "TaskStateCollector",
    "ToolResultMessage",
    "trim_to_budget",
    "UserMessage",
]
