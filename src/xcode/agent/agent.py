"""Agent 薄封装类。

给 run_agent_loop() 加实例状态管理和更友好的调用接口。
不重新实现循环逻辑，内部委托给 run_agent_loop()。
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable
from typing import Any

from .agent_loop import run_agent_loop
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopResult,
    AgentMessage,
    AgentTool,
    CancellationSignal,
)


class Agent:
    """纯 agent 运行时薄封装。

    持有工具列表、steer/followup 队列。
    不感知 ToolSpec、权限、审计、hook — 这些由调用方通过
    AgentLoopConfig 的钩子注入。
    """

    def __init__(
        self,
        tools: list[AgentTool[Any]],
    ) -> None:
        self._tools = tools
        self._steer_queue: list[AgentMessage] = []
        self._followup_queue: list[AgentMessage] = []
        self._last_result: AgentLoopResult | None = None

    # ── 队列 API ──

    def steer(self, msg: AgentMessage) -> None:
        """向 steer 队列注入消息（下一轮循环开始前消费）。"""
        self._steer_queue.append(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        """向 followup 队列注入消息（当前循环结束后追加）。"""
        self._followup_queue.append(msg)

    # ── 执行 ──

    async def run(
        self,
        messages: list[AgentMessage],
        config: AgentLoopConfig,
        *,
        signal: CancellationSignal | None = None,
        emit: Callable[[AgentEvent], None] | None = None,
    ) -> AgentLoopResult:
        """执行 agent 循环，返回结果。

        config 每次调用传入，不缓存。队列 drain 逻辑自动注入。
        """
        effective = self._inject_queues(config)
        context = AgentContext(tools=list(self._tools))
        sink = emit or (lambda _e: None)
        result = await run_agent_loop(messages, context, effective, sink, signal)
        self._last_result = result
        return result

    async def run_stream(
        self,
        messages: list[AgentMessage],
        config: AgentLoopConfig,
        *,
        signal: CancellationSignal | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """执行 agent 循环，以异步迭代器实时产出事件。

        事件在 run_agent_loop 执行过程中通过 asyncio.Queue 实时传递，
        消费方可边跑边 yield。run_agent_loop 抛出的异常会传播给消费方。
        """
        effective = self._inject_queues(config)
        context = AgentContext(tools=list(self._tools))
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        error_slot: BaseException | None = None

        def _emit(event: AgentEvent) -> None:
            queue.put_nowait(event)

        async def _run() -> None:
            nonlocal error_slot
            try:
                result = await run_agent_loop(
                    messages, context, effective, _emit, signal
                )
                self._last_result = result
            except BaseException as exc:
                error_slot = exc
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            if error_slot is not None:
                raise error_slot
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ── 内部 ──

    def _drain_steer_all(self) -> list[AgentMessage]:
        msgs = list(self._steer_queue)
        self._steer_queue.clear()
        return msgs

    def _drain_followup_all(self) -> list[AgentMessage]:
        msgs = list(self._followup_queue)
        self._followup_queue.clear()
        return msgs

    def _pop_steer_one(self) -> list[AgentMessage]:
        if self._steer_queue:
            return [self._steer_queue.pop(0)]
        return []

    def _pop_followup_one(self) -> list[AgentMessage]:
        if self._followup_queue:
            return [self._followup_queue.pop(0)]
        return []

    def _inject_queues(self, config: AgentLoopConfig) -> AgentLoopConfig:
        """将队列 drain 逻辑注入 config，返回新实例。"""
        steer_fn = (
            self._pop_steer_one
            if config.steering_mode == "one-at-a-time"
            else self._drain_steer_all
        )
        followup_fn = (
            self._pop_followup_one
            if config.follow_up_mode == "one-at-a-time"
            else self._drain_followup_all
        )
        return dataclasses.replace(
            config,
            get_steering_messages=steer_fn,
            get_follow_up_messages=followup_fn,
        )
