"""AI 层类型定义：核心类型与 provider 配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import orjson

KnownApi = Literal[
    "openai-completions",
    "anthropic-messages",
    "deepseek-chat",
    "mimo-chat",
    "google-gemini",
]

Api = KnownApi | str

KnownProvider = Literal[
    "anthropic",
    "openai",
    "deepseek",
    "mimo",
    "google",
    "azure",
]

Provider = KnownProvider | str

type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
type ReasoningSummary = Literal["auto", "concise", "detailed"]
ModelThinkingLevel = ThinkingLevel | str
Transport = Literal["sse", "websocket", "auto"]
CacheRetention = Literal["none", "short", "long"]
PromptCacheRetention = Literal["in_memory", "24h"]
ServiceTier = Literal["auto", "default", "flex", "scale", "priority"]
TextVerbosity = Literal["low", "medium", "high"]
Truncation = Literal["auto", "disabled"]
type ToolArguments = dict[str, object]


@dataclass(frozen=True)
class ProviderConfig:
    """标准化 provider 构造参数。

    所有 provider 共享此构造契约，factory 无需感知 provider 专有字段。
    专有配置通过 extra 传递，各 provider 在 __init__ 中自行提取。
    """

    api_key: str
    model: str
    base_url: str = ""
    context_window: int | None = None
    thinking: bool = True
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Cost:
    """LLM 调用成本（美元/百万 token）。

    支持分时计价：peak_hours 声明高峰时段（UTC 小时，含头不含尾），
    为空表示全天固定费率；off_peak_factor 为高峰以外的费率系数。
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0
    peak_hours: tuple[tuple[int, int], ...] = ()
    off_peak_factor: float = 1.0

    def effective(self, now: datetime | None = None) -> Cost:
        """按调用时刻返回有效费率：高峰时段用全价，否则按 off_peak_factor 折算。"""
        if not self.peak_hours or self._is_peak_hour(now or datetime.now(timezone.utc)):
            return self
        return Cost(
            input=self.input * self.off_peak_factor,
            output=self.output * self.off_peak_factor,
            cache_read=self.cache_read * self.off_peak_factor,
            cache_write=self.cache_write * self.off_peak_factor,
            total=self.total * self.off_peak_factor,
            peak_hours=self.peak_hours,
            off_peak_factor=self.off_peak_factor,
        )

    def _is_peak_hour(self, now: datetime) -> bool:
        hour = now.hour
        return any(start <= hour < end for start, end in self.peak_hours)


@dataclass(frozen=True)
class Usage:
    """LLM 调用用量。"""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)


@dataclass(frozen=True)
class Model:
    """模型元数据。"""

    id: str
    name: str
    api: str
    provider: str
    base_url: str = ""
    reasoning: bool = False
    context_window: int = 0
    max_tokens: int = 0
    cost: Cost = field(default_factory=Cost)
    thinking_level_map: dict[str, str | None] | None = None


@dataclass(frozen=True)
class ThinkingBudgets:
    """Extended thinking token 预算配置。

    用于支持 extended thinking 的模型（如 o1/o3/DeepSeek R1），
    控制各思考级别的最大 token 数。字段含义：
    - minimal: 最简思考（快速响应）
    - low: 低强度思考
    - medium: 中等强度思考
    - high: 高强度思考（深度推理）
    - xhigh: 极高强度思考（深度研究）
    """

    minimal: int = 0
    low: int = 0
    medium: int = 0
    high: int = 0
    xhigh: int = 0


@dataclass(frozen=True)
class StreamOptions:
    """单次 provider 请求的可选覆盖参数。"""

    temperature: float | None = None
    max_tokens: int | None = None
    signal: Any | None = None
    api_key: str | None = None
    transport: Transport = "auto"
    cache_retention: CacheRetention = "short"
    session_id: str | None = None
    reasoning: str | None = None
    reasoning_summary: ReasoningSummary | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any | None = None
    on_response: Any | None = None
    thinking_budgets: ThinkingBudgets | None = None
    thinking_level: str | None = None
    tool_choice: str | dict[str, Any] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    user: str | None = None
    response_extra_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """LLM 可见的工具 schema。"""

    name: str
    description: str
    parameters: dict[str, Any]
    builtin: dict[str, Any] | None = None


# --- Context serialization ---


def dump_context(
    system_prompt: str | None,
    messages: list[dict[str, Any]],
) -> str:
    obj: dict[str, Any] = {"messages": messages}
    if system_prompt:
        obj["system_prompt"] = system_prompt
    return orjson.dumps(obj, default=str).decode()


def load_context(data: str) -> tuple[str | None, list[dict[str, Any]]]:
    obj = orjson.loads(data.encode())
    messages = obj.get("messages", [])
    system_prompt: str | None = obj.get("system_prompt")
    return system_prompt, messages
