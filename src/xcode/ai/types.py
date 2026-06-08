from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import orjson

"""AI 层类型定义：Model、Transport、Thinking、Usage 等核心类型。"""

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
ModelThinkingLevel = ThinkingLevel | str
Transport = Literal["sse", "websocket", "auto"]
CacheRetention = Literal["none", "short", "long"]


@dataclass(frozen=True)
class Cost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)


@dataclass(frozen=True)
class Model[TApi: Api]:
    id: str
    name: str
    api: TApi
    provider: str
    base_url: str = ""
    reasoning: bool = False
    context_window: int = 0
    max_tokens: int = 0
    cost: Cost = field(default_factory=Cost)
    thinking_level_map: dict[str, str | None] | None = None


@dataclass
class ThinkingBudgets:
    """Extended thinking token 预算配置。

    用于支持 extended thinking 的模型（如 o1/o3/DeepSeek R1），
    控制各思考级别的最大 token 数。字段含义：
    - minimal: 最简思考（快速响应）
    - low: 低强度思考
    - medium: 中等强度思考
    - high: 高强度思考（深度推理）
    """

    minimal: int = 0
    low: int = 0
    medium: int = 0
    high: int = 0


@dataclass(frozen=True)
class StreamOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    signal: Any | None = None
    api_key: str | None = None
    transport: Transport = "auto"
    cache_retention: CacheRetention = "short"
    session_id: str | None = None
    reasoning: str | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    on_payload: Any | None = None
    on_response: Any | None = None
    thinking_budgets: ThinkingBudgets | None = None
    thinking_level: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """LLM 可见的工具 schema。"""

    name: str
    description: str
    schema: dict[str, Any]


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
