from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from typing import Literal

from blinker import Signal

from ..session import JsonValue
from .correlation import EventCorrelation

"""类型安全的执行框架事件钩子系统。"""

logger = logging.getLogger("xcode.harness.observability.hooks")

HookEvent = Literal[
    "pre_tool",
    "post_tool",
    "on_error",
    "on_compact",
    "before_agent_start",
    "before_provider_request",
]
HookCallback = Callable[["HookRecord"], None]
HarnessCallback = Callable[["HarnessEvent"], None]
type HookMetadata = dict[str, JsonValue]

_HOOK_EVENTS: tuple[HookEvent, ...] = (
    "pre_tool",
    "post_tool",
    "on_error",
    "on_compact",
    "before_agent_start",
    "before_provider_request",
)


@dataclass(frozen=True)
class HookRecord:
    event: HookEvent
    tool: str = ""
    input: str = ""
    output: str = ""
    error: str = ""
    metadata: Mapping[str, object] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = ""
    turn_id: str = ""
    request_id: str = ""
    tool_call_id: str = ""


@dataclass(frozen=True)
class PreToolEvent:
    """工具执行前事件。"""

    type: Literal["pre_tool"] = "pre_tool"
    tool: str = ""
    input: str = ""
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class PostToolEvent:
    """工具执行后事件。"""

    type: Literal["post_tool"] = "post_tool"
    tool: str = ""
    input: str = ""
    output: str = ""
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class ErrorEvent:
    """工具或运行期错误事件。"""

    type: Literal["on_error"] = "on_error"
    tool: str = ""
    input: str = ""
    error: str = ""
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class CompactEvent:
    """上下文压缩事件。"""

    type: Literal["on_compact"] = "on_compact"
    metadata: HookMetadata = field(default_factory=dict)
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class BeforeAgentStartEvent:
    """Agent 启动前事件。"""

    type: Literal["before_agent_start"] = "before_agent_start"
    question: str = ""
    mode: str = "act"
    metadata: HookMetadata = field(default_factory=dict)
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


@dataclass(frozen=True)
class BeforeProviderRequestEvent:
    """Provider 请求前事件。"""

    type: Literal["before_provider_request"] = "before_provider_request"
    messages: list[HookMetadata] = field(default_factory=list)
    tools: list[HookMetadata] = field(default_factory=list)
    metadata: HookMetadata = field(default_factory=dict)
    correlation: EventCorrelation = field(default_factory=EventCorrelation)


type HarnessEvent = (
    PreToolEvent
    | PostToolEvent
    | ErrorEvent
    | CompactEvent
    | BeforeAgentStartEvent
    | BeforeProviderRequestEvent
)


class HookManager:
    def __init__(self) -> None:
        self._registered: dict[HookEvent, Signal] = {
            event: Signal() for event in _HOOK_EVENTS
        }
        self._subscribed: dict[HookEvent, Signal] = {
            event: Signal() for event in _HOOK_EVENTS
        }

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        self._registered[event].connect(callback, weak=False)

    def remove(self, event: HookEvent, callback: HookCallback) -> None:
        self._registered[event].disconnect(callback)

    def subscribe(self, event: HookEvent, callback: HarnessCallback) -> None:
        self._subscribed[event].connect(callback, weak=False)

    def unsubscribe(self, event: HookEvent, callback: HarnessCallback) -> None:
        self._subscribed[event].disconnect(callback)

    def emit(self, record: HookRecord) -> None:
        self._registered[record.event].send(record)
        converted = _harness_event_from_hook(record)
        self._subscribed[record.event].send(converted)


def _harness_event_from_hook(record: HookRecord) -> HarnessEvent:
    metadata = _hook_metadata(record.metadata)
    correlation = EventCorrelation(
        timestamp=record.timestamp,
        session_id=record.session_id,
        turn_id=record.turn_id,
        request_id=record.request_id,
        tool_call_id=record.tool_call_id,
    )

    def _pre_tool() -> PreToolEvent:
        return PreToolEvent(
            tool=record.tool,
            input=record.input,
            correlation=correlation,
        )

    def _post_tool() -> PostToolEvent:
        return PostToolEvent(
            tool=record.tool,
            input=record.input,
            output=record.output,
            correlation=correlation,
        )

    def _on_error() -> ErrorEvent:
        return ErrorEvent(
            tool=record.tool,
            input=record.input,
            error=record.error,
            correlation=correlation,
        )

    def _on_compact() -> CompactEvent:
        return CompactEvent(metadata=metadata, correlation=correlation)

    def _before_agent_start() -> BeforeAgentStartEvent:
        return BeforeAgentStartEvent(
            question=str(metadata.get("question", "")),
            mode=str(metadata.get("mode", "act")),
            metadata=metadata,
            correlation=correlation,
        )

    def _before_provider_request() -> BeforeProviderRequestEvent:
        return BeforeProviderRequestEvent(
            messages=_hook_metadata_list(metadata.get("messages")),
            tools=_hook_metadata_list(metadata.get("tools")),
            metadata=metadata,
            correlation=correlation,
        )

    _DISPATCH: dict[str, Callable[[], HarnessEvent]] = {
        "pre_tool": _pre_tool,
        "post_tool": _post_tool,
        "on_error": _on_error,
        "on_compact": _on_compact,
        "before_agent_start": _before_agent_start,
        "before_provider_request": _before_provider_request,
    }
    builder = _DISPATCH.get(record.event)
    return (
        builder()
        if builder is not None
        else CompactEvent(metadata=metadata, correlation=correlation)
    )


def _hook_metadata(value: object) -> HookMetadata:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_value(item) for key, item in value.items()}


def _hook_metadata_list(value: object) -> list[HookMetadata]:
    if not isinstance(value, list):
        return []
    return [_hook_metadata(item) for item in value if isinstance(item, Mapping)]


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
