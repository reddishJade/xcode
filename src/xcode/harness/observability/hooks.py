from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Any, Literal

from blinker import Signal

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
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreToolEvent:
    """工具执行前事件。"""

    type: Literal["pre_tool"] = "pre_tool"
    tool: str = ""
    input: str = ""


@dataclass(frozen=True)
class PostToolEvent:
    """工具执行后事件。"""

    type: Literal["post_tool"] = "post_tool"
    tool: str = ""
    input: str = ""
    output: str = ""


@dataclass(frozen=True)
class ErrorEvent:
    """工具或运行期错误事件。"""

    type: Literal["on_error"] = "on_error"
    tool: str = ""
    input: str = ""
    error: str = ""


@dataclass(frozen=True)
class CompactEvent:
    """上下文压缩事件。"""

    type: Literal["on_compact"] = "on_compact"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeforeAgentStartEvent:
    """Agent 启动前事件。"""

    type: Literal["before_agent_start"] = "before_agent_start"
    question: str = ""
    mode: str = "act"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeforeProviderRequestEvent:
    """Provider 请求前事件。"""

    type: Literal["before_provider_request"] = "before_provider_request"
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def emit(self, record: HookRecord) -> list[Any]:
        results: list[Any] = [
            rv for _receiver, rv in self._registered[record.event].send(record)
        ]
        converted = _harness_event_from_hook(record)
        results.extend(
            rv for _receiver, rv in self._subscribed[record.event].send(converted)
        )
        return results


def _harness_event_from_hook(record: HookRecord) -> HarnessEvent:
    metadata = record.metadata or {}
    if record.event == "pre_tool":
        return PreToolEvent(tool=record.tool, input=record.input)
    if record.event == "post_tool":
        return PostToolEvent(
            tool=record.tool,
            input=record.input,
            output=record.output,
        )
    if record.event == "on_error":
        return ErrorEvent(tool=record.tool, input=record.input, error=record.error)
    if record.event == "on_compact":
        return CompactEvent(metadata=metadata)
    if record.event == "before_agent_start":
        return BeforeAgentStartEvent(
            question=str(metadata.get("question", "")),
            mode=str(metadata.get("mode", "act")),
            metadata=metadata,
        )
    if record.event == "before_provider_request":
        return BeforeProviderRequestEvent(
            messages=_dict_list(metadata.get("messages")),
            tools=_dict_list(metadata.get("tools")),
            metadata=metadata,
        )
    return CompactEvent(metadata=metadata)


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
