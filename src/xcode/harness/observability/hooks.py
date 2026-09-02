from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from blinker import Signal

from ..session import JsonValue
from .correlation import EventCorrelation

"""类型安全的执行框架事件钩子系统。"""

logger = logging.getLogger("xcode.harness.observability.hooks")

HookEvent = Literal[
    "pre_tool",
    "post_tool",
    "on_error",
    "on_context_window_reset",
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
    "on_context_window_reset",
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
class ContextWindowResetHookEvent:
    """上下文窗口切换事件。"""

    type: Literal["on_context_window_reset"] = "on_context_window_reset"
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
    | ContextWindowResetHookEvent
    | BeforeAgentStartEvent
    | BeforeProviderRequestEvent
)


class HookManager(Protocol):
    def register(self, event: HookEvent, callback: HookCallback) -> None: ...
    def remove(self, event: HookEvent, callback: HookCallback) -> None: ...
    def register_background(self, event: HookEvent, callback: HookCallback) -> None: ...
    def remove_background(self, event: HookEvent, callback: HookCallback) -> None: ...
    def subscribe(self, event: HookEvent, callback: HarnessCallback) -> None: ...
    def unsubscribe(self, event: HookEvent, callback: HarnessCallback) -> None: ...
    def emit(self, record: HookRecord) -> None: ...
    def drain_background(self) -> None: ...


class SignalHookManager:
    def __init__(self) -> None:
        self._registered: dict[HookEvent, Signal] = {
            event: Signal() for event in _HOOK_EVENTS
        }
        self._subscribed: dict[HookEvent, Signal] = {
            event: Signal() for event in _HOOK_EVENTS
        }
        self._background: dict[HookEvent, Signal] = {
            event: Signal() for event in _HOOK_EVENTS
        }
        self._background_queue: queue.Queue[tuple[HookEvent, HookRecord]] = (
            queue.Queue()
        )
        self._worker_started = False
        self._worker_lock = threading.Lock()

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        self._registered[event].connect(callback, weak=False)

    def remove(self, event: HookEvent, callback: HookCallback) -> None:
        self._registered[event].disconnect(callback)

    def register_background(self, event: HookEvent, callback: HookCallback) -> None:
        self._background[event].connect(callback, weak=False)

    def remove_background(self, event: HookEvent, callback: HookCallback) -> None:
        self._background[event].disconnect(callback)

    def subscribe(self, event: HookEvent, callback: HarnessCallback) -> None:
        self._subscribed[event].connect(callback, weak=False)

    def unsubscribe(self, event: HookEvent, callback: HarnessCallback) -> None:
        self._subscribed[event].disconnect(callback)

    def emit(self, record: HookRecord) -> None:
        self._registered[record.event].send(record)
        converted = _harness_event_from_hook(record)
        self._subscribed[record.event].send(converted)
        if self._background[record.event].receivers:
            self._ensure_worker()
            self._background_queue.put((record.event, record))

    def drain_background(self) -> None:
        """等待已入队后台 hook 完成，供测试和受控关闭使用。"""
        self._background_queue.join()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker_started:
                return
            thread = threading.Thread(
                target=self._run_background,
                name="xcode-hook-manager",
                daemon=True,
            )
            thread.start()
            self._worker_started = True

    def _run_background(self) -> None:
        while True:
            event, record = self._background_queue.get()
            try:
                self._send_background(event, record)
            finally:
                self._background_queue.task_done()

    def _send_background(self, event: HookEvent, record: HookRecord) -> None:
        for receiver in self._background[event].receivers_for(record):
            try:
                receiver(record)
            except Exception:
                logger.exception("background hook failed for event %s", event)


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

    def _on_context_window_reset() -> ContextWindowResetHookEvent:
        return ContextWindowResetHookEvent(
            metadata=metadata,
            correlation=correlation,
        )

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
        "on_context_window_reset": _on_context_window_reset,
        "before_agent_start": _before_agent_start,
        "before_provider_request": _before_provider_request,
    }
    builder = _DISPATCH.get(record.event)
    return (
        builder()
        if builder is not None
        else ContextWindowResetHookEvent(
            metadata=metadata,
            correlation=correlation,
        )
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
