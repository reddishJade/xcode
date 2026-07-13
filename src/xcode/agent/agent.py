"""Agent 薄封装类。

给 run_agent_loop() 加实例状态管理和更友好的调用接口。
不重新实现循环逻辑，内部委托给 run_agent_loop()。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from threading import Lock

from xcode.ai.providers.base import ModelProvider

from .agent_loop import run_agent_loop
from .config import AgentContext, AgentLoopConfig
from ._codec import convert_to_llm
from .results import AgentLoopResult
from .events import (
    AgentEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from .messages import AgentMessage, AssistantMessage, SystemMessage, UserMessage
from .types import AgentTool, CancellationSignal, TextContent


def _tool_update_label(tool_name: str, args: dict[str, object]) -> str:
    if tool_name == "read_file":
        path = str(args.get("path") or args.get("file_path") or args.get("input") or "")
        return f"read {path}" if path else tool_name
    if tool_name == "bash":
        command = str(args.get("command") or args.get("input") or "")
        return f"$ {command}" if command else tool_name
    if tool_name == "grep_search":
        pattern = str(
            args.get("pattern") or args.get("query") or args.get("input") or ""
        )
        path = str(args.get("path") or args.get("include") or "workspace")
        return f"grep /{pattern}/ in {path}" if pattern else tool_name
    if tool_name in {"glob_files", "find_files"}:
        pattern = str(
            args.get("pattern") or args.get("path") or args.get("input") or ""
        )
        path = str(args.get("path") or "workspace")
        return f"find {pattern} in {path}" if pattern else tool_name
    return tool_name


class Agent:
    """纯 agent 运行时薄封装。

    持有工具列表、steer/followup 队列。
    不感知 ToolSpec、权限、审计、hook — 这些由调用方通过
    AgentLoopConfig 的钩子注入。

    当提供了 model + system_prompt，可通过 prompt() 快速执行（subagent 场景）。
    """

    def __init__(
        self,
        tools: list[AgentTool],
        *,
        model: ModelProvider | None = None,
        system_prompt: str = "",
    ) -> None:
        self._tools = tools
        self._model = model
        self._system_prompt = system_prompt
        self._steer_queue: list[AgentMessage] = []
        self._steer_lock = Lock()
        self._accepting_steer = True
        self._followup_queue: list[AgentMessage] = []
        self._last_result: AgentLoopResult | None = None
        self._last_messages: list[AgentMessage] = []

    # ── 队列 API ──

    def steer(self, msg: AgentMessage) -> bool:
        """向 steer 队列注入消息（下一轮循环开始前消费）。

        设计原因：
        steer 用于循环内中断和调整方向，消息在下一步开始前插入。
        这允许外部代码在工具执行后、模型调用前干预（如注入上下文）。
        """
        return self.try_steer(msg)

    def try_steer(self, msg: AgentMessage) -> bool:
        """仅在当前 run 仍有消费边界时接受 steer 消息。"""
        with self._steer_lock:
            if not self._accepting_steer:
                return False
            self._steer_queue.append(msg)
            return True

    def _drain_steer_queue(self) -> list[AgentMessage]:
        """原子地取出等待在下一个循环边界消费的 steer 消息。"""
        with self._steer_lock:
            messages = self._steer_queue
            self._steer_queue = []
        return messages

    def _finish_steering(self) -> list[AgentMessage]:
        """关闭 steer 入口并原子取出末轮消息。"""
        with self._steer_lock:
            self._accepting_steer = False
            messages = self._steer_queue
            self._steer_queue = []
        return messages

    def _reopen_steering(self) -> None:
        """末轮收到 steer 后重新开放下一模型边界。"""
        with self._steer_lock:
            self._accepting_steer = True

    def close_steering(self) -> list[AgentMessage]:
        """结束 run 时关闭入口并返回尚未消费的消息。"""
        return self._finish_steering()

    def _continue_with(self, msg: AgentMessage) -> None:
        """向 same-run continuation 队列注入消息。

        设计原因：
        followup 用于循环后续任务，消息在当前循环自然结束后追加。
        这允许外部代码安排下一轮工作（如多阶段任务编排）。
        """
        self._followup_queue.append(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        """兼容旧接口；该方法不是 next-run follow-up。"""
        self._continue_with(msg)

    def update_tools(self, tools: list[AgentTool]) -> None:
        """替换当前工具列表。

        用于执行模式切换等场景，需动态更换可用工具集。
        """
        self._tools = tools

    @property
    def last_result(self) -> AgentLoopResult | None:
        return self._last_result

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._last_messages)

    async def prompt(
        self,
        text: str,
        *,
        model: ModelProvider | None = None,
        system_prompt: str | None = None,
        signal: CancellationSignal | None = None,
        on_update: Callable[[str], None] | None = None,
    ) -> str:
        """轻量 prompt API：创建简化 AgentLoopConfig 并执行。

        用于 subagent 等自包含场景。不依赖 hooks / permissions / compaction。
        """
        model = model or self._model
        if model is None:
            raise ValueError("model is required for prompt()")
        sp = system_prompt if system_prompt is not None else self._system_prompt
        history: list[AgentMessage] = [SystemMessage(content=sp)] if sp else []
        config = AgentLoopConfig(
            provider=model,
            max_steps=25,
            convert_to_llm=convert_to_llm,
        )

        def _emit(event: AgentEvent) -> None:
            if on_update is None:
                return
            if isinstance(event, ToolExecutionStartEvent):
                on_update(f"→ {_tool_update_label(event.tool_name, event.args)}")
            elif isinstance(event, ToolExecutionEndEvent):
                status = "✓" if not event.is_error else "✗"
                on_update(f"{status} {event.tool_name}")

        result = await self.run(
            [UserMessage(content=text)],
            config,
            signal=signal,
            history=history,
            emit=_emit if on_update else None,
        )
        self._last_messages = result.messages
        parts: list[str] = []
        for msg in result.messages:
            if not isinstance(msg, AssistantMessage):
                continue
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    parts.append(block.text)
        return " ".join(parts).strip() or "(no output)"

    # ── 执行 ──

    async def run(
        self,
        messages: list[AgentMessage],
        config: AgentLoopConfig,
        *,
        signal: CancellationSignal | None = None,
        emit: Callable[[AgentEvent], None] | None = None,
        history: list[AgentMessage] | None = None,
    ) -> AgentLoopResult:
        """执行 agent 循环，返回结果。

        config 和队列引用每次调用传入，不缓存。
        """
        context = AgentContext(
            messages=list(history or []),
            tools=list(self._tools),
        )
        sink = emit or (lambda _e: None)
        result = await run_agent_loop(
            messages,
            context,
            config,
            sink,
            signal,
            steer_queue=self._drain_steer_queue,
            finish_steering=self._finish_steering,
            reopen_steering=self._reopen_steering,
            follow_up_queue=self._followup_queue,
        )
        self._last_result = result
        return result

    async def run_stream(
        self,
        messages: list[AgentMessage],
        config: AgentLoopConfig,
        *,
        signal: CancellationSignal | None = None,
        history: list[AgentMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """执行 agent 循环，以异步迭代器实时产出事件。

        事件在 run_agent_loop 执行过程中通过 asyncio.Queue 实时传递，
        消费方可边跑边 yield。run_agent_loop 抛出的异常会传播给消费方。
        """
        context = AgentContext(
            messages=list(history or []),
            tools=list(self._tools),
        )
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        error_slot: BaseException | None = None

        def _emit(event: AgentEvent) -> None:
            queue.put_nowait(event)

        async def _run() -> None:
            nonlocal error_slot
            try:
                result = await run_agent_loop(
                    messages,
                    context,
                    config,
                    _emit,
                    signal,
                    steer_queue=self._drain_steer_queue,
                    finish_steering=self._finish_steering,
                    reopen_steering=self._reopen_steering,
                    follow_up_queue=self._followup_queue,
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
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            # error_slot 在此处抛出，因为 finally 是生成器退出前最后执行的代码
            if error_slot is not None:
                raise error_slot
