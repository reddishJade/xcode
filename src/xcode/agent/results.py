"""Agent 循环指标和结果类型。

从 config.py 提取，与配置和上下文类型分离。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import SkipValidation

from xcode.ai.events import ProviderFailure
from xcode.ai.providers.base import StreamProvider

from .messages import AgentMessage


class TerminationReason(StrEnum):
    """Agent 循环的统一终止原因。"""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    STEP_LIMIT = "step_limit"
    WATCHDOG = "watchdog"
    PROVIDER_ERROR = "provider_error"


class AgentLoopMetrics(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    model_latencies_ms: list[float] = Field(default_factory=list)
    tool_latencies_ms: list[float] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_config = ConfigDict(extra="forbid")


class AgentLoopResult(BaseModel):
    messages: list[AgentMessage] = Field(default_factory=list)
    surface: list[AgentMessage]
    steps: int = 0
    termination_reason: TerminationReason = TerminationReason.COMPLETED
    watchdog_reason: str | None = None
    error_detail: str | None = None
    provider_failure: ProviderFailure | None = None
    metrics: AgentLoopMetrics | None = None
    active_provider: Annotated[StreamProvider | None, SkipValidation] = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
