from __future__ import annotations


import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import queue
from time import perf_counter
from typing import Any, AsyncIterator, Generator, Iterator, cast

from ...agent.messages import convert_to_llm
from ...agent.provider_response import provider_events_to_response
from ...agent.types import (
    AgentMessage,
    AssistantMessage,
    ContentBlock,
    SystemMessage,
    TextContent,
    ToolCallBlock,
    UserMessage,
)

from .cancellation import CancellationToken
from .compaction import (
    CompactController,
    budget_large_tool_outputs,
    estimate_message_tokens,
    estimate_text_tokens,
    latest_read_file_tool_result_ids,
)
from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    ToolCall as ToolUseBlock,
)
from xcode.ai.providers.protocol import ModelProvider
from xcode.harness.adapters.tool_schema import tool_definitions_from_specs
from .tool_events import ToolResult
from .execution_modes import mode_notice, policy_for_mode
from .tool_executor import (
    ExecutionCancelled,
    ToolExecutor,
    stringify_tool_input,
    tool_result_message,
)
from ..config import AgentConfig, ExecutionMode
from ..observability import AuditRecord, HookManager, HookRecord, PermissionPolicy
from ..skills import ApprovalCallback, ToolSpec
from .async_worker import IsolatedAsyncWorker

"""结构化工具调用执行框架。

本模块把模型响应看作 content blocks：text、tool_use、tool_result，并在
事件流中暴露模型输出、工具调用、工具结果和最终答案。
"""

__all__ = [
    "StructuredAgent",
    "StructuredAgentEvent",
    "StructuredAgentResult",
]

StructuredCompactor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
RuntimeContextProvider = Callable[[str], list[str]]


def _to_dict(msg: AgentMessage) -> dict[str, Any]:
    """将类型化消息转为 dict（保持 state.messages 的 dict 格式）。"""
    result = convert_to_llm([msg])
    assert result, f"convert_to_llm returned empty for {type(msg).__name__}"
    return result[0]


def _blocks_to_typed(blocks: list[dict[str, Any]]) -> list[ContentBlock]:
    """将 raw dict content blocks 转为类型化 ContentBlock。"""
    result: list[ContentBlock] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            result.append(TextContent(text=str(b.get("text", ""))))
        elif b.get("type") == "tool_use":
            result.append(
                ToolCallBlock(
                    id=str(b.get("id", "")),
                    name=str(b.get("name", "")),
                    arguments=b.get("input", {}),
                )
            )
    return result


def _typed_blocks_to_raw(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """将 Agent ContentBlock 转为 StructuredAgent 运行状态使用的 raw block。"""
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallBlock):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments or {},
                }
            )
    return result


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    status: str = "ok"
    type: str = "tool_result"


@dataclass(frozen=True)
class StructuredAgentResult:
    answer: str
    messages: list[dict[str, Any]]
    steps: int
    tool_calls: list[ToolUseBlock]
    stopped_by_limit: bool = False
    metrics: dict[str, Any] | None = None
    stopped_by_watchdog: bool = False
    watchdog_reason: str | None = None
    needs_follow_up: bool = False  # 是否需要后续工具调用步骤


@dataclass(frozen=True)
class StructuredAgentEvent:
    type: str
    step: int
    data: Any


class RunState:
    """Agent runtime state to clear scattered variables in structured.py."""

    def __init__(self, messages: list[dict[str, Any]], mode: ExecutionMode) -> None:
        self.messages = messages
        self.mode = mode
        self.tool_calls: list[ToolUseBlock] = []
        self.last_tool_signature: str | None = None
        self.repeated_tool_count: int = 0
        self.step: int = 1
        self.consecutive_continuations: int = 0
        self.consecutive_idle_steps: int = 0  # 连续空转步数计数器
        self.step_retries: int = 0  # 当前步的 session 级重试次数
        self.metrics: dict[str, Any] = {
            "llm_calls": 0,
            "tool_calls": 0,
            "estimated_prompt_tokens": 0,
            "model_latencies_ms": [],
            "tool_latencies_ms": [],
        }


class StructuredAgent:
    """与 provider 解耦的结构化工具调用循环。"""

    def __init__(
        self,
        provider: ModelProvider,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        compactor: StructuredCompactor | None = None,
        manual_compact_requested: Callable[[], bool] | None = None,
        compact_controller: CompactController | None = None,
        audit_logger: Callable[[AuditRecord], None] | None = None,
        session_id: str = "local",
        permission_policy: PermissionPolicy | None = None,
        hook_manager: HookManager | None = None,
        runtime_context_provider: RuntimeContextProvider | None = None,
        cancellation_token: CancellationToken | None = None,
        speculation_planner: Any = None,
        fallback_provider: ModelProvider | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.provider = provider
        self.fallback_provider = fallback_provider  # 备选故障转移 provider
        self.project_root = project_root
        self.registry = registry
        self.tool_map = {tool.name: tool for tool in registry}
        self.config = config or AgentConfig()
        self.approval_callback = approval_callback
        self.compactor = compactor
        self.manual_compact_requested = manual_compact_requested or (
            compact_controller.consume if compact_controller else None
        )
        self._compact_controller = compact_controller
        self.audit_logger = audit_logger
        self.session_id = session_id

        # 加载 settings.json 安全沙箱
        effective_policy = permission_policy
        if project_root is not None:
            local_settings = project_root / ".local" / "settings.json"
            root_settings = project_root / "settings.json"
            settings_path = (
                local_settings
                if local_settings.exists()
                else (root_settings if root_settings.exists() else None)
            )
            if settings_path is not None:
                from ..observability.permissions import (
                    SettingsSandboxPermissionPolicy,
                    CompositePermissionPolicy,
                )

                sandbox = SettingsSandboxPermissionPolicy(settings_path)
                effective_policy = cast(
                    PermissionPolicy,
                    CompositePermissionPolicy(sandbox, permission_policy),
                )

        self.permission_policy = effective_policy
        self.hook_manager = hook_manager
        self.runtime_context_provider = runtime_context_provider
        self.cancellation_token = cancellation_token or CancellationToken()
        self.speculation_planner = speculation_planner
        self._consecutive_errors: int = 0
        self.steer_queue: list[AgentMessage] = []
        self.followup_queue: list[AgentMessage] = []

    def steer(self, msg: AgentMessage) -> None:
        """在下一轮对话开始前注入消息（steer）。"""
        self.steer_queue.append(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        """在当前轮对话后追加消息（follow-up）。"""
        self.followup_queue.append(msg)

    def request_compaction(self) -> None:
        """主动请求进行上下文压缩（在下一次模型调用前生效）。"""
        if self._compact_controller is not None:
            self._compact_controller.request()

    def run(
        self,
        question: str,
        mode: ExecutionMode | None = None,
    ) -> StructuredAgentResult:
        return _run_coro_sync(self.arun(question, mode=mode))

    async def run_async(
        self,
        question: str,
        mode: ExecutionMode | None = None,
    ) -> StructuredAgentResult:
        return await self.arun(question, mode=mode)

    async def arun(
        self,
        question: str,
        mode: ExecutionMode | None = None,
    ) -> StructuredAgentResult:
        result: StructuredAgentResult | None = None
        async for event in self.arun_stream(question, mode=mode):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(
        self,
        question: str,
        mode: ExecutionMode | None = None,
    ) -> Iterator[StructuredAgentEvent]:
        yield from _aiter_to_sync_iter(
            self.arun_stream(question, mode=mode), self.cancellation_token
        )

    async def arun_stream(
        self,
        question: str,
        mode: ExecutionMode | None = None,
    ) -> AsyncIterator[StructuredAgentEvent]:
        effective_mode = mode or self.config.execution_mode
        policy = policy_for_mode(effective_mode)
        active_registry = policy.filter_tools(self.registry)
        active_tool_map = {tool.name: tool for tool in active_registry}
        self.cancellation_token.reset()
        messages: list[dict[str, Any]] = self._initial_messages(
            question,
            effective_mode,
        )
        state = RunState(messages, effective_mode)

        for step in range(1, self.config.max_steps + 1):
            state.step = step
            if self.cancellation_token.is_cancelled():
                yield StructuredAgentEvent(
                    "final",
                    state.step,
                    StructuredAgentResult(
                        answer=self.cancellation_token.reason,
                        messages=state.messages,
                        steps=state.step,
                        tool_calls=state.tool_calls,
                    ),
                )
                return

            # 注入 steer 消息（下一轮开始前）
            while self.steer_queue:
                steer_msg = self.steer_queue.pop(0)
                state.messages.append(_to_dict(steer_msg))

            accumulated_blocks = []
            while True:
                if self._should_compact(state.messages):
                    self._emit_hook(
                        HookRecord(
                            "on_compact", metadata={"messages": len(state.messages)}
                        )
                    )
                    assert self.compactor is not None
                    state.messages = self.compactor(state.messages)
                state.messages = _budget_messages_for_provider(state.messages)
                state.metrics["llm_calls"] += 1
                state.metrics["estimated_prompt_tokens"] += estimate_message_tokens(
                    state.messages
                )
                started = perf_counter()
                (
                    response_blocks,
                    stop_reason,
                    stream_events,
                    reasoning_content,
                ) = await self._call_model_streaming_async(
                    state.messages,
                    state.step,
                    active_registry,
                )
                state.metrics["model_latencies_ms"].append(_elapsed_ms(started))
                for event in stream_events:
                    yield event

                if stop_reason == "error":
                    # session 级别自动重试（指数退避）
                    retries = state.step_retries + 1
                    state.step_retries = retries
                    if retries <= 3:
                        delay = 0.5 * (2 ** (retries - 1))
                        await asyncio.sleep(delay)
                        continue
                    state.messages.append(
                        _to_dict(
                            AssistantMessage(
                                content=_blocks_to_typed(response_blocks),
                                reasoning_content=_reasoning_for_assistant(
                                    response_blocks, reasoning_content
                                ),
                            )
                        )
                    )
                    yield StructuredAgentEvent("assistant", state.step, response_blocks)
                    accumulated_blocks.extend(response_blocks)

                    if not accumulated_blocks:
                        yield StructuredAgentEvent(
                            "final",
                            state.step,
                            StructuredAgentResult(
                                answer="I encountered an error."
                                if not state.messages
                                else "I encountered an error. Please try again.",
                                messages=state.messages,
                                steps=state.step,
                                tool_calls=state.tool_calls,
                            ),
                        )
                        return
                    break

                if stop_reason == "max_tokens":
                    inc = estimate_text_tokens(
                        json.dumps(response_blocks, ensure_ascii=False, default=str)
                    )
                    if inc < 500:
                        state.consecutive_continuations += 1
                    else:
                        state.consecutive_continuations = 0

                    if state.consecutive_continuations >= 3:
                        raise RuntimeError(
                            "Diminishing Returns: Consecutive output token increments below 500 limit reached."
                        )

                    state.messages.append(
                        _to_dict(
                            AssistantMessage(
                                content=_blocks_to_typed(response_blocks),
                                reasoning_content=_reasoning_for_assistant(
                                    response_blocks, reasoning_content
                                ),
                            )
                        )
                    )
                    yield StructuredAgentEvent("assistant", state.step, response_blocks)
                    accumulated_blocks.extend(response_blocks)

                    state.messages.append(_to_dict(UserMessage(content="continue")))
                    continue
                else:
                    state.consecutive_continuations = 0
                    state.messages.append(
                        _to_dict(
                            AssistantMessage(
                                content=_blocks_to_typed(response_blocks),
                                reasoning_content=_reasoning_for_assistant(
                                    response_blocks, reasoning_content
                                ),
                            )
                        )
                    )
                    yield StructuredAgentEvent("assistant", state.step, response_blocks)
                    accumulated_blocks.extend(response_blocks)
                    break

            uses = [
                _to_tool_use(block)
                for block in accumulated_blocks
                if _is_tool_use(block)
            ]
            if not uses:
                yield StructuredAgentEvent(
                    "final",
                    state.step,
                    StructuredAgentResult(
                        answer=_text_from_blocks(accumulated_blocks),
                        messages=state.messages,
                        steps=state.step,
                        tool_calls=state.tool_calls,
                        metrics=_finalize_metrics(state.metrics),
                        needs_follow_up=False,
                    ),
                )
                return
            watchdog_reason, state.last_tool_signature, state.repeated_tool_count = (
                _check_repeated_tool_watchdog(
                    uses,
                    state.last_tool_signature,
                    state.repeated_tool_count,
                    self.config.watchdog_repeated_tool_limit,
                )
            )
            if watchdog_reason is not None:
                yield StructuredAgentEvent(
                    "final",
                    state.step,
                    StructuredAgentResult(
                        answer=watchdog_reason,
                        messages=state.messages,
                        steps=state.step,
                        tool_calls=state.tool_calls,
                        metrics=_finalize_metrics(state.metrics),
                        stopped_by_watchdog=True,
                        watchdog_reason=watchdog_reason,
                        needs_follow_up=False,
                    ),
                )
                return

            for tool_use in uses:
                state.tool_calls.append(tool_use)
                yield StructuredAgentEvent("tool_use", state.step, tool_use)

            results = await self._execute_tool_uses(
                uses=uses,
                step=state.step,
                active_tool_map=active_tool_map,
                mode=state.mode,
                policy=policy,
            )
            for tool_use, result in zip(uses, results, strict=True):
                state.metrics["tool_calls"] += 1
                if result.elapsed_ms is not None:
                    state.metrics["tool_latencies_ms"].append(result.elapsed_ms)
                yield StructuredAgentEvent(
                    "tool_result",
                    state.step,
                    ToolResultBlock(result.tool_call_id, result.content, result.status),
                )
                for event in self._emit_speculation(
                    tool_use.name, result.status, state.step
                ):
                    yield event
            state.messages.append(
                {
                    "role": "user",
                    "content": [tool_result_message(result) for result in results],
                }
            )

            # 语义空转熔断 (Multi-step Idle Failsafe) - 仅在 Act Mode 下执行
            is_productive = False
            for tool_use, result in zip(uses, results, strict=True):
                if result.status == "ok":
                    tool_spec = active_tool_map.get(tool_use.name)
                    if tool_spec and tool_spec.read_only:
                        is_productive = True
                        break
                    if tool_use.name in (
                        "write_file",
                        "edit_file",
                        "write_to_file",
                        "replace_file_content",
                        "multi_replace_file_content",
                    ):
                        is_productive = True
                        break
                    if tool_use.name == "bash":
                        is_productive = True
                        break

            if is_productive:
                state.consecutive_idle_steps = 0
            else:
                if state.mode == "act":
                    state.consecutive_idle_steps += 1

            if state.consecutive_idle_steps >= 4:
                raise RuntimeError(
                    "Watchdog triggered: 4 consecutive steps without any successful file writes or shell command executions."
                )

            # 注入 follow-up 消息（当前工具步骤执行完后追加，继续下一轮）
            while self.followup_queue:
                fu_msg = self.followup_queue.pop(0)
                state.messages.append(_to_dict(fu_msg))

        yield StructuredAgentEvent(
            "final",
            self.config.max_steps,
            StructuredAgentResult(
                answer="step limit reached",
                messages=state.messages,
                steps=self.config.max_steps,
                tool_calls=state.tool_calls,
                stopped_by_limit=True,
                metrics=_finalize_metrics(state.metrics),
                needs_follow_up=False,
            ),
        )

    def _call_model_streaming(
        self,
        messages: list[dict[str, Any]],
        step: int,
        registry: tuple[ToolSpec, ...],
    ) -> tuple[
        list[dict[str, Any]], str | None, list[StructuredAgentEvent], str | None
    ]:
        return _run_coro_sync(
            self._call_model_streaming_async(messages, step, registry)
        )

    async def _call_model_streaming_async(
        self,
        messages: list[dict[str, Any]],
        step: int,
        registry: tuple[ToolSpec, ...],
    ) -> tuple[
        list[dict[str, Any]], str | None, list[StructuredAgentEvent], str | None
    ]:
        events = await self._provider_events(messages, registry)

        response = provider_events_to_response(events)
        stream_events = [
            StructuredAgentEvent(f"{delta.kind}_delta", step, delta.chunk)
            for delta in response.deltas
        ]
        return (
            _typed_blocks_to_raw(response.content),
            response.stop_reason,
            stream_events,
            response.reasoning_content,
        )

    async def _provider_events(
        self,
        messages: list[dict[str, Any]],
        registry: tuple[ToolSpec, ...],
    ) -> list[ProviderEvent]:
        if self.provider is None:
            return [FinalMessage("StructuredAgent requires a provider", "error")]

        import random

        max_retries = 3
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                events = await _collect_provider_events(
                    self.provider, messages, registry
                )
                self._consecutive_errors = 0
                return events
            except Exception as e:
                last_error = e
                name = type(e).__name__
                msg = str(e).lower()
                is_transient = (
                    "ratelimit" in name.lower()
                    or "429" in msg
                    or "overloaded" in name.lower()
                    or "529" in msg
                    or "overloaded" in msg
                )
                if not is_transient:
                    return [FinalMessage(f"Provider error: {e}", "error")]

                self._consecutive_errors += 1

                if self._consecutive_errors >= 3 and self.fallback_provider is not None:
                    self.provider = self.fallback_provider
                    self._consecutive_errors = 0
                    try:
                        return await _collect_provider_events(
                            self.provider, messages, registry
                        )
                    except Exception as e2:
                        return [FinalMessage(f"Fallback provider error: {e2}", "error")]

                if attempt >= max_retries:
                    return [
                        FinalMessage(f"Provider repeatedly unavailable: {e}", "error")
                    ]

                base_delay = 0.5 * (2**attempt)
                base = min(base_delay, 32.0)
                jitter = random.uniform(0, base * 0.25)
                delay = base + jitter
                await asyncio.sleep(delay)
        return [FinalMessage(f"Provider unavailable: {last_error}", "error")]

    async def _execute_tool_uses(
        self,
        *,
        uses: list[ToolUseBlock],
        step: int,
        active_tool_map: dict[str, ToolSpec],
        mode: ExecutionMode,
        policy,
    ) -> list[ToolResult]:
        cancel = asyncio.Event()
        if self.cancellation_token.is_cancelled():
            cancel.set()
        executor = ToolExecutor(
            self.registry,
            tool_workers=self.config.tool_workers,
            approval_callback=self.approval_callback,
            permission_policy=self.permission_policy,
            hook_manager=self.hook_manager,
            audit_logger=self.audit_logger,
            session_id=self.session_id,
            policy=policy,
        )
        try:
            return await executor.execute(
                uses,
                cancel=cancel,
                active_tool_map=active_tool_map,
                mode=mode,
            )
        except ExecutionCancelled:
            return [
                ToolResult(tool_use.id, self.cancellation_token.reason, "interrupted")
                for tool_use in uses
            ]

    def _emit_hook(self, record: HookRecord) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(record)

    def _should_compact(self, messages: list[dict[str, Any]]) -> bool:
        if self.compactor is None:
            return False
        if (
            self.manual_compact_requested is not None
            and self.manual_compact_requested()
        ):
            return True
        return (
            self.config.compact_threshold > 0
            and len(messages) > self.config.compact_threshold
        ) or (
            self.config.compact_token_threshold > 0
            and estimate_message_tokens(messages) > self.config.compact_token_threshold
        )

    def _initial_messages(
        self,
        question: str,
        mode: ExecutionMode = "act",
    ) -> list[dict[str, Any]]:
        """创建初始消息列表（AgentMessage 构造后转为 dict）。"""
        typed_messages: list[AgentMessage] = []
        mode_notice_text = mode_notice(mode)
        if self.runtime_context_provider is not None:
            context_parts = self.runtime_context_provider(question)
            if mode_notice_text:
                context_parts.append(mode_notice_text)
            if context_parts:
                typed_messages.append(
                    SystemMessage(
                        content="\n\n".join(part for part in context_parts if part)
                    )
                )
        elif mode_notice_text:
            typed_messages.append(SystemMessage(content=mode_notice_text))
        typed_messages.append(UserMessage(content=question))
        return convert_to_llm(typed_messages)

    def _emit_speculation(
        self,
        tool_name: str | None,
        status: str,
        step: int,
    ) -> Generator[StructuredAgentEvent, None, None]:
        if self.speculation_planner is None:
            return
        event = self.speculation_planner.plan(tool_name, status)
        if event is not None:
            yield StructuredAgentEvent("speculation", step, event)


def _is_tool_use(block: dict[str, Any]) -> bool:
    return block.get("type") == "tool_use"


def _to_tool_use(block: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(
        id=str(block.get("id", "")),
        name=str(block.get("name", "")),
        input=block.get("input", {}),
    )


def _text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif "text" in block:
            parts.append(str(block["text"]))
    return "".join(parts).strip()


def _reasoning_for_assistant(
    blocks: list[dict[str, Any]],
    reasoning_content: str | None,
) -> str | None:
    if reasoning_content is not None:
        return reasoning_content
    if any(_is_tool_use(block) for block in blocks):
        return ""
    return None


def _budget_messages_for_provider(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """发送模型前裁剪过大的非文件读取工具结果。"""
    preserved_tool_results = latest_read_file_tool_result_ids(messages)
    return budget_large_tool_outputs(
        messages,
        large_tool_output_chars=8_000,
        large_tool_output_head_chars=4_000,
        large_tool_output_tail_chars=4_000,
        compact_token_threshold=1,
        budget_trigger_token_ratio=0,
        preserve_tool_result_ids=preserved_tool_results,
    )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(metrics)
    finalized["model_total_ms"] = round(sum(metrics["model_latencies_ms"]), 3)
    finalized["tool_total_ms"] = round(sum(metrics["tool_latencies_ms"]), 3)
    finalized["total_observed_ms"] = round(
        finalized["model_total_ms"] + finalized["tool_total_ms"],
        3,
    )
    return finalized


def _check_repeated_tool_watchdog(
    uses: list[ToolUseBlock],
    last_signature: str | None,
    repeated_count: int,
    limit: int,
) -> tuple[str | None, str | None, int]:
    if limit <= 0 or len(uses) != 1:
        return None, None, 0
    signature = f"{uses[0].name}:{stringify_tool_input(uses[0].input)}"
    if signature == last_signature:
        repeated_count += 1
    else:
        repeated_count = 1
    if repeated_count > limit:
        return (
            f"watchdog stopped repeated tool call: {uses[0].name}",
            signature,
            repeated_count,
        )
    return None, signature, repeated_count


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if hasattr(coro, "close"):
        coro.close()
    raise RuntimeError(
        "StructuredAgent.run() cannot be called inside an active event loop; "
        "use await StructuredAgent.run_async() instead."
    )


def _aiter_to_sync_iter(
    async_iter: AsyncIterator[StructuredAgentEvent],
    cancellation_token: CancellationToken,
) -> Iterator[StructuredAgentEvent]:
    items: queue.Queue[tuple[str, Any]] = queue.Queue()
    worker = IsolatedAsyncWorker(name="xcode-sync-stream-worker")

    async def consume() -> None:
        try:
            async for event in async_iter:
                items.put(("item", event))
        except BaseException as exc:
            items.put(("error", exc))
        finally:
            items.put(("done", None))

    future = worker.submit(consume())
    try:
        while True:
            kind, payload = items.get()
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                return
    finally:
        if not future.done():
            cancellation_token.cancel("sync stream consumer stopped")
            future.cancel()
        worker.close()


async def _collect_provider_events(
    provider: ModelProvider,
    messages: list[dict[str, Any]],
    registry: tuple[ToolSpec, ...],
) -> list[ProviderEvent]:
    events = []
    tool_definitions = tool_definitions_from_specs(registry)
    async for event in provider.stream(messages, tool_definitions):
        events.append(event)
    return events
