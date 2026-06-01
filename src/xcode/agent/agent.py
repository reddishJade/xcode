from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..harness.agent_runtime.cancellation import CancellationToken
from .agent_loop import run_agent_loop, run_agent_loop_continue
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AssistantMessage,
    ImageContent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    ThinkingLevel,
    ToolExecutionMode,
    TurnEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    UserMessage,
)

EMPTY_USAGE = {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "cache_write": 0,
    "total_tokens": 0,
}

DEFAULT_MODEL_NAME = "unknown"


# ── 消息队列 ──


class PendingMessageQueue:
    def __init__(self, mode: str = "one-at-a-time") -> None:
        self.messages: list[AgentMessage] = []
        self.mode = mode

    def enqueue(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def has_items(self) -> bool:
        return len(self.messages) > 0

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = list(self.messages)
            self.messages.clear()
            return drained
        if not self.messages:
            return []
        first = self.messages[0]
        self.messages = self.messages[1:]
        return [first]

    def clear(self) -> None:
        self.messages.clear()


# ── Active Run ──


class ActiveRun:
    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.cancellation_token = CancellationToken()


# ── Agent ──


class Agent:
    """有状态的 Agent 封装。"""

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Any = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool[Any]] | None = None,
        messages: list[AgentMessage] | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], list[dict[str, Any]]]
        | None = None,
        transform_context: Callable[[list[AgentMessage], Any], list[AgentMessage]]
        | None = None,
        get_api_key: Callable[[str], str | None] | None = None,
        before_tool_call: Callable[[Any, Any], Any] | None = None,
        after_tool_call: Callable[[Any, Any], Any] | None = None,
        prepare_next_turn: Callable[[], Any] | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        session_id: str | None = None,
        tool_execution: ToolExecutionMode = "parallel",
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._thinking_level = thinking_level
        self._tools = list(tools) if tools else []
        self._messages = list(messages) if messages else []

        self._is_streaming = False
        self._streaming_message: AgentMessage | None = None
        self._pending_tool_calls: set[str] = set()
        self._error_message: str | None = None

        self.listeners: list[Callable[[AgentEvent, Any], None]] = []
        self.steering_queue = PendingMessageQueue(steering_mode)
        self.follow_up_queue = PendingMessageQueue(follow_up_mode)

        self.convert_to_llm = convert_to_llm
        self.transform_context = transform_context
        self.get_api_key = get_api_key
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.prepare_next_turn = prepare_next_turn
        self.session_id = session_id
        self.tool_execution = tool_execution

        self._active_run: ActiveRun | None = None

    # ── 属性 ──

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    @property
    def tools(self) -> list[AgentTool[Any]]:
        return list(self._tools)

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def streaming_message(self) -> AgentMessage | None:
        return self._streaming_message

    @property
    def pending_tool_calls(self) -> set[str]:
        return set(self._pending_tool_calls)

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    # ── 队列 ──

    def steer(self, message: AgentMessage) -> None:
        self.steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        self.follow_up_queue.enqueue(message)

    def clear_queues(self) -> None:
        self.steering_queue.clear()
        self.follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        return self.steering_queue.has_items() or self.follow_up_queue.has_items()

    # ── 生命周期 ──

    def add_listener(self, listener: Callable[[AgentEvent, Any], None]) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[AgentEvent, Any], None]) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def abort(self) -> None:
        if self._active_run:
            self._active_run.cancellation_token.cancel()

    async def wait_for_idle(self) -> None:
        if self._active_run:
            await self._active_run.done.wait()

    def reset(self) -> None:
        self._messages = []
        self._is_streaming = False
        self._streaming_message = None
        self._pending_tool_calls = set()
        self._error_message = None
        self.clear_queues()

    # ── 主入口 ──

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
        **kwargs,
    ) -> None:
        if self._active_run:
            raise RuntimeError("Agent is already processing a prompt.")
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        if self._active_run:
            raise RuntimeError("Agent is already processing.")

        last = self._messages[-1] if self._messages else None
        if not last:
            raise ValueError("No messages to continue from")

        if isinstance(last, AssistantMessage):
            queued = self.steering_queue.drain()
            if queued:
                await self._run_prompt_messages(queued)
                return

            queued = self.follow_up_queue.drain()
            if queued:
                await self._run_prompt_messages(queued)
                return

            raise ValueError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # ── 内部 ──

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return input
        if not isinstance(input, str):
            return [input]

        content: list[TextContent | ImageContent] = [TextContent(text=input)]
        if images:
            content.extend(images)
        return [UserMessage(content=content)]

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._system_prompt,
            messages=list(self._messages),
            tools=list(self._tools),
        )

    def _build_loop_config(self) -> AgentLoopConfig:
        return AgentLoopConfig(
            model=self._model,
            tool_execution=self.tool_execution,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            prepare_next_turn=self.prepare_next_turn,
            get_steering_messages=lambda: self.steering_queue.drain(),
            get_follow_up_messages=lambda: self.follow_up_queue.drain(),
        )

    async def _run_prompt_messages(self, messages: list[AgentMessage]) -> None:
        context = self._create_context_snapshot()
        config = self._build_loop_config()

        async def executor(signal: CancellationToken) -> None:
            await run_agent_loop(
                prompts=messages,
                context=context,
                config=config,
                emit=self._emit,
                signal=signal,
                stream_fn=None,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        context = self._create_context_snapshot()
        config = self._build_loop_config()

        async def executor(signal: CancellationToken) -> None:
            await run_agent_loop_continue(
                context=context,
                config=config,
                emit=self._emit,
                signal=signal,
                stream_fn=None,
            )

        await self._run_with_lifecycle(executor)

    def _emit(self, event: AgentEvent) -> None:
        """同步 emit 函数，供 agent_loop 调用。"""
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
            self._streaming_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._streaming_message = None
            if event.message is not None:
                self._messages.append(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            self._pending_tool_calls.add(event.tool_call_id)
        elif isinstance(event, ToolExecutionEndEvent):
            self._pending_tool_calls.discard(event.tool_call_id)
        elif isinstance(event, TurnEndEvent):
            if (
                isinstance(event.message, AssistantMessage)
                and event.message.error_message
            ):
                self._error_message = event.message.error_message
        elif isinstance(event, AgentEndEvent):
            self._streaming_message = None

        for listener in self.listeners:
            try:
                result = listener(event, None)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                pass

    async def _run_with_lifecycle(self, executor: Callable[[Any], Any]) -> None:
        if self._active_run:
            raise RuntimeError("Agent is already processing.")

        active = ActiveRun()
        self._active_run = active
        self._is_streaming = True
        self._streaming_message = None
        self._error_message = None

        try:
            await executor(active.cancellation_token)
        except Exception as e:
            self._handle_run_failure(e, False)
        finally:
            self._finish_run()

    def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        msg = AssistantMessage(
            content=[],
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
        )
        self._emit(MessageStartEvent(message=msg))
        self._emit(MessageEndEvent(message=msg))
        self._emit(TurnEndEvent(message=msg, tool_results=[]))
        self._emit(AgentEndEvent(messages=[msg]))

    def _finish_run(self) -> None:
        self._is_streaming = False
        self._streaming_message = None
        self._pending_tool_calls = set()
        if self._active_run:
            self._active_run.done.set()
            self._active_run = None


