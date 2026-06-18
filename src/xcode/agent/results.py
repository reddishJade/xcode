"""Agent 循环指标和结果类型。

从 config.py 提取，与配置和上下文类型分离。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from xcode.ai.providers.protocol import StreamProvider


class TerminationReason(StrEnum):
    """Agent 循环的统一终止原因。"""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    STEP_LIMIT = "step_limit"
    WATCHDOG = "watchdog"
    PROVIDER_ERROR = "provider_error"


@dataclass
class AgentLoopMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    model_latencies_ms: list[float] = field(default_factory=list)
    tool_latencies_ms: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentLoopResult:
    messages: list = field(default_factory=list)
    steps: int = 0
    termination_reason: TerminationReason = TerminationReason.COMPLETED
    watchdog_reason: str | None = None
    error_detail: str | None = None
    metrics: AgentLoopMetrics | None = None
    active_provider: StreamProvider | None = None

    @property
    def stopped_by_limit(self) -> bool:
        """兼容旧调用方；停止状态以 termination_reason 为准。"""
        return self.termination_reason is TerminationReason.STEP_LIMIT

    @property
    def stopped_by_watchdog(self) -> bool:
        """兼容旧调用方；停止状态以 termination_reason 为准。"""
        return self.termination_reason is TerminationReason.WATCHDOG

    @property
    def stopped_by_error(self) -> bool:
        """兼容旧调用方；停止状态以 termination_reason 为准。"""
        return self.termination_reason is TerminationReason.PROVIDER_ERROR
