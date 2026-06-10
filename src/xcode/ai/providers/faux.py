from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    ReasoningDelta,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    UsageUpdate,
)
from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import StreamOptions, ToolDefinition

"""模拟 provider：可脚本化响应队列，用于测试。"""


# --- Faux response builders ---


def faux_text(text: str) -> list[ProviderEvent]:
    """构建纯文本响应的事件列表。"""
    return [
        TextDelta(chunk=text),
        FinalMessage(content=text, stop_reason="end_turn"),
    ]


def faux_thinking(text: str) -> list[ProviderEvent]:
    """构建 thinking 文本的事件列表。"""
    return [ReasoningDelta(chunk=text)]


def faux_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    tool_id: str | None = None,
) -> list[ProviderEvent]:
    """构建单次工具调用的事件列表。"""
    tid = tool_id or f"faux_{name}"
    return [
        ToolCallEvent(calls=[ToolCall(id=tid, name=name, input=arguments or {})]),
        FinalMessage(content="", stop_reason="tool_use"),
    ]


def faux_usage(input_tokens: int = 10, output_tokens: int = 20) -> list[ProviderEvent]:
    return [UsageUpdate(input_tokens=input_tokens, output_tokens=output_tokens)]


def faux_final(stop_reason: StopReason = "end_turn") -> list[ProviderEvent]:
    return [FinalMessage(content="", stop_reason=stop_reason)]


# --- Response queue provider ---


@dataclass
class FauxResponse:
    events: list[ProviderEvent]
    delay_seconds: float = 0.0


class FauxProvider(ModelProvider):
    """可脚本化的模拟 provider。

    支持预设多轮响应队列、tool call、thinking、usage 事件。
    每次 ``stream()`` 消费队列中的一个响应。

    接受：
    - ProviderEvent 列表（单轮）
    - ProviderEvent 列表的列表（多轮）
    - ``(messages, tools) -> list[ProviderEvent]`` 工厂函数
    """

    def __init__(
        self,
        response_spec: Any = None,
        *,
        model: str = "faux-model",
        delay_seconds: float = 0.0,
    ) -> None:
        self.model = model
        self._delay = delay_seconds
        self._queue: list[FauxResponse] = []
        self.call_count = 0
        self._last_messages: list[dict[str, Any]] = []
        self._last_tools: list[ToolDefinition] = []
        self._factory: Any = None

        if response_spec is not None:
            if callable(response_spec):
                self._factory = response_spec
            else:
                items = list(response_spec)
                if (
                    items
                    and isinstance(items[0], Sequence)
                    and not isinstance(
                        items[0],
                        (
                            FinalMessage,
                            ReasoningDelta,
                            TextDelta,
                            ToolCallEvent,
                            UsageUpdate,
                        ),
                    )
                ):
                    for resp in items:
                        self._queue.append(FauxResponse(events=list(resp)))
                else:
                    self._queue.append(
                        FauxResponse(
                            events=list(response_spec), delay_seconds=delay_seconds
                        )
                    )

    def push(self, events: list[ProviderEvent], delay: float = 0.0) -> None:
        """向队列尾部添加一个响应。"""
        self._queue.append(FauxResponse(events=events, delay_seconds=delay))

    def push_text(self, text: str) -> None:
        self.push(faux_text(text))

    def push_tool_call(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> None:
        self.push(faux_tool_call(name, arguments))

    def clear(self) -> None:
        self._queue.clear()
        self.call_count = 0

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        self.call_count += 1
        self._last_messages = messages
        self._last_tools = tools

        if not self._queue:
            if self._factory is not None:
                for event in self._factory(messages, tools):
                    yield event
                return
            yield FinalMessage(
                content="No more faux responses queued.",
                stop_reason="error",
            )
            return

        resp = self._queue.pop(0)
        if resp.delay_seconds > 0:
            await asyncio.sleep(resp.delay_seconds)
        for event in resp.events:
            yield event

    @property
    def last_messages(self) -> list[dict[str, Any]]:
        return self._last_messages

    @property
    def last_tools(self) -> list[ToolDefinition]:
        return self._last_tools


def register_faux_provider(
    responses: list[list[ProviderEvent]] | None = None,
    *,
    model: str = "faux-model",
    model_id: str = "faux",
    delay_seconds: float = 0.0,
) -> FauxProvider:
    """快捷创建 FauxProvider。"""
    return FauxProvider(
        response_spec=responses,
        model=model,
        delay_seconds=delay_seconds,
    )
