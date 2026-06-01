from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

"""类型安全的执行框架事件钩子系统。"""

HookEvent = Literal[
    "pre_tool",
    "post_tool",
    "on_error",
    "on_compact",
    "before_agent_start",
    "before_provider_request",
]
HookCallback = Callable[["HookRecord"], None]


@dataclass(frozen=True)
class HookRecord:
    event: HookEvent
    tool: str = ""
    input: str = ""
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] | None = None


@dataclass
class BeforeAgentStartHook:
    question: str = ""
    mode: str = "act"


@dataclass
class BeforeProviderRequestHook:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)


class HookManager:
    def __init__(self) -> None:
        self._callbacks: dict[HookEvent, list[HookCallback]] = defaultdict(list)

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        """注册钩子。"""
        self._callbacks[event].append(callback)

    def remove(self, event: HookEvent, callback: HookCallback) -> None:
        """移除钩子。"""
        try:
            self._callbacks[event].remove(callback)
        except ValueError:
            return

    def emit(self, record: HookRecord) -> list[Any]:
        results: list[Any] = []
        for callback in self._callbacks.get(record.event, []):
            try:
                result = callback(record)
                results.append(result)
            except Exception:
                pass
        return results
