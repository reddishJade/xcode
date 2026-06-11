"""Agent 循环指标和结果类型。

从 config.py 提取，与配置和上下文类型分离。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xcode.ai.providers.protocol import StreamProvider


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
    stopped_by_limit: bool = False
    stopped_by_watchdog: bool = False
    stopped_by_error: bool = False
    watchdog_reason: str | None = None
    metrics: AgentLoopMetrics | None = None
    active_provider: StreamProvider | None = None
