"""Xcode Agent — 类型化 Agent 循环。"""

from __future__ import annotations

from .agent import Agent
from .agent_loop import run_agent_loop
from .config import AgentContext, AgentLoopConfig
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
    "run_agent_loop",
    "SystemMessage",
    "ToolResultMessage",
    "UserMessage",
]
