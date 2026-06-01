from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
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
    minimal: int = 0
    low: int = 0
    medium: int = 0
    high: int = 0


@dataclass(frozen=True)
class ProviderResponse:
    status: int
    headers: dict[str, str]


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


# Content block types
@dataclass(frozen=True)
class TextContent:
    type: str = "text"
    text: str = ""


@dataclass(frozen=True)
class ImageContent:
    type: str = "image"
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCallContent:
    type: str = "tool_call"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThinkingContent:
    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None


@dataclass(frozen=True)
class ToolResultContent:
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    status: str = "ok"
