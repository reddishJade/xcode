"""Agent 薄封装类。

给 run_agent_loop() 加实例状态管理和更友好的调用接口。
不重新实现循环逻辑，内部委托给 run_agent_loop()。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from xcode.ai.providers.base import ModelProvider

from .agent_loop import run_agent_loop
from .config import AgentContext, AgentLoopConfig
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

    持有工具列表；step 输入的所有权由调用方注入。
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
        self._last_result: AgentLoopResult | None = None
        self._last_messages: list[AgentMessage] = []

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
        loop_config: AgentLoopConfig | None = None,
        signal: CancellationSignal | None = None,
        on_update: Callable[[str], None] | None = None,
    ) -> str:
        """轻量 prompt API：创建或使用调用方提供的 AgentLoopConfig 执行。

        subagent 等需要权限边界的调用方必须显式注入 hooks。
        """
        model = model or self._model
        if model is None:
            raise ValueError("model is required for prompt()")
        sp = system_prompt if system_prompt is not None else self._system_prompt
        request_prefix: list[AgentMessage] = [SystemMessage(content=sp)] if sp else []
        config = loop_config or AgentLoopConfig(provider=model)
        config_updates: dict[str, object] = {}
        if config.provider is None:
            config_updates["provider"] = model
        if config_updates:
            config = config.model_copy(update=config_updates)

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
            request_prefix=request_prefix,
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
        request_prefix: list[AgentMessage] | None = None,
        step_input: Callable[[], list[AgentMessage]] | None = None,
        finish_step_input: Callable[[], list[AgentMessage]] | None = None,
        reopen_step_input: Callable[[], None] | None = None,
    ) -> AgentLoopResult:
        """执行 agent 循环，返回结果。

        config 和队列引用每次调用传入，不缓存。
        """
        context = AgentContext(
            request_prefix=list(request_prefix or []),
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
            steer_queue=step_input,
            finish_steering=finish_step_input,
            reopen_steering=reopen_step_input,
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
        request_prefix: list[AgentMessage] | None = None,
        step_input: Callable[[], list[AgentMessage]] | None = None,
        finish_step_input: Callable[[], list[AgentMessage]] | None = None,
        reopen_step_input: Callable[[], None] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """执行 agent 循环，以异步迭代器实时产出事件。

        事件在 run_agent_loop 执行过程中通过 asyncio.Queue 实时传递，
        消费方可边跑边 yield。run_agent_loop 抛出的异常会传播给消费方。
        """
        context = AgentContext(
            request_prefix=list(request_prefix or []),
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
                    steer_queue=step_input,
                    finish_steering=finish_step_input,
                    reopen_steering=reopen_step_input,
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
