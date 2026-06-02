"""结构化工具调用执行框架。

本模块把模型响应看作 content blocks：text、tool_use、tool_result，并在
事件流中暴露模型输出、工具调用、工具结果和最终答案。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator, Generator, Iterator, cast

from ...agent.messages import convert_to_llm
from ...agent.types import AgentMessage, AssistantMessage, SystemMessage, UserMessage
from .agent_helpers import (
    budget_messages_for_provider,
    blocks_to_typed,
    check_repeated_tool_watchdog,
    elapsed_ms,
    finalize_metrics,
    is_tool_use,
    reasoning_for_assistant,
    run_coro_sync,
    aiter_to_sync_iter,
    text_from_blocks,
    to_dict,
    to_tool_use,
)

# re-export for test backward compatibility
_budget_messages_for_provider = budget_messages_for_provider
_check_repeated_tool_watchdog = check_repeated_tool_watchdog
_finalize_metrics = finalize_metrics
_elapsed_ms = elapsed_ms
from .cancellation import CancellationToken
from .compaction import CompactController, estimate_message_tokens, estimate_text_tokens
from xcode.ai.events import ToolCall as ToolUseBlock
from xcode.ai.providers.protocol import ModelProvider
from .tool_events import ToolResult
from .execution_modes import mode_notice, policy_for_mode
from .streaming import call_model_streaming, execute_tool_uses
from .tool_executor import tool_result_message
from ..config import AgentConfig, ExecutionMode
from ..observability import AuditRecord, HookManager, HookRecord, PermissionPolicy
from ..skills import ApprovalCallback, ToolSpec

__all__ = ["StructuredAgent", "StructuredAgentEvent", "StructuredAgentResult"]

StructuredCompactor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
RuntimeContextProvider = Callable[[str], list[str]]


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
    needs_follow_up: bool = False


@dataclass(frozen=True)
class StructuredAgentEvent:
    type: str
    step: int
    data: Any


class RunState:
    def __init__(self, messages: list[dict[str, Any]], mode: ExecutionMode) -> None:
        self.messages = messages
        self.mode = mode
        self.tool_calls: list[ToolUseBlock] = []
        self.last_tool_signature: str | None = None
        self.repeated_tool_count: int = 0
        self.step: int = 1
        self.consecutive_continuations: int = 0
        self.consecutive_idle_steps: int = 0
        self.step_retries: int = 0
        self.consecutive_errors: int = 0
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
        self.fallback_provider = fallback_provider
        self.project_root = project_root
        self.registry = registry
        self.tool_map = {t.name: t for t in registry}
        self.config = config or AgentConfig()
        self.approval_callback = approval_callback
        self.compactor = compactor
        self.manual_compact_requested = manual_compact_requested or (
            compact_controller.consume if compact_controller else None
        )
        self._compact_controller = compact_controller
        self.audit_logger = audit_logger
        self.session_id = session_id
        self.permission_policy = _resolve_permission_policy(
            project_root, permission_policy
        )
        self.hook_manager = hook_manager
        self.runtime_context_provider = runtime_context_provider
        self.cancellation_token = cancellation_token or CancellationToken()
        self.speculation_planner = speculation_planner
        self._consecutive_errors: int = 0
        self.steer_queue: list[AgentMessage] = []
        self.followup_queue: list[AgentMessage] = []

    # ── 公共 API ──

    def steer(self, msg: AgentMessage) -> None:
        self.steer_queue.append(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        self.followup_queue.append(msg)

    def request_compaction(self) -> None:
        if self._compact_controller is not None:
            self._compact_controller.request()

    def run(
        self, question: str, mode: ExecutionMode | None = None
    ) -> StructuredAgentResult:
        return run_coro_sync(self.arun(question, mode=mode))

    async def run_async(
        self, question: str, mode: ExecutionMode | None = None
    ) -> StructuredAgentResult:
        return await self.arun(question, mode=mode)

    async def arun(
        self, question: str, mode: ExecutionMode | None = None
    ) -> StructuredAgentResult:
        result: StructuredAgentResult | None = None
        async for event in self.arun_stream(question, mode=mode):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> Iterator[StructuredAgentEvent]:
        yield from aiter_to_sync_iter(
            self.arun_stream(question, mode=mode), self.cancellation_token
        )

    async def arun_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AsyncIterator[StructuredAgentEvent]:
        effective_mode = mode or self.config.execution_mode
        policy = policy_for_mode(effective_mode)
        active_registry = policy.filter_tools(self.registry)
        active_tool_map = {t.name: t for t in active_registry}
        self.cancellation_token.reset()
        state = RunState(
            self._initial_messages(question, effective_mode), effective_mode
        )

        for step in range(1, self.config.max_steps + 1):
            state.step = step
            if self.cancellation_token.is_cancelled():
                yield _final_event(
                    state.step,
                    StructuredAgentResult(
                        answer=self.cancellation_token.reason,
                        messages=state.messages,
                        steps=state.step,
                        tool_calls=state.tool_calls,
                    ),
                )
                return

            while self.steer_queue:
                state.messages.append(to_dict(self.steer_queue.pop(0)))

            inner_events: list[StructuredAgentEvent] = []
            accumulated = await self._run_inner_loop(
                state, active_registry, inner_events
            )
            for e in inner_events:
                yield e
            if accumulated is None:
                return

            uses = [to_tool_use(b) for b in accumulated if is_tool_use(b)]
            if not uses:
                yield _final_event(
                    state.step,
                    StructuredAgentResult(
                        answer=text_from_blocks(accumulated),
                        messages=state.messages,
                        steps=state.step,
                        tool_calls=state.tool_calls,
                        metrics=finalize_metrics(state.metrics),
                    ),
                )
                return

            wd_reason, state.last_tool_signature, state.repeated_tool_count = (
                check_repeated_tool_watchdog(
                    uses,
                    state.last_tool_signature,
                    state.repeated_tool_count,
                    self.config.watchdog_repeated_tool_limit,
                )
            )
            if wd_reason is not None:
                yield _final_event(
                    state.step,
                    StructuredAgentResult(
                        answer=wd_reason,
                        messages=state.messages,
                        steps=state.step,
                        tool_calls=state.tool_calls,
                        metrics=finalize_metrics(state.metrics),
                        stopped_by_watchdog=True,
                        watchdog_reason=wd_reason,
                    ),
                )
                return

            for tu in uses:
                state.tool_calls.append(tu)
                yield StructuredAgentEvent("tool_use", state.step, tu)

            results = await execute_tool_uses(
                uses=uses,
                registry=self.registry,
                tool_workers=self.config.tool_workers,
                approval_callback=self.approval_callback,
                permission_policy=self.permission_policy,
                hook_manager=self.hook_manager,
                audit_logger=self.audit_logger,
                session_id=self.session_id,
                policy=policy,
                cancellation_token=self.cancellation_token,
                active_tool_map=active_tool_map,
                mode=state.mode,
            )
            for tu, res in zip(uses, results, strict=True):
                state.metrics["tool_calls"] += 1
                if res.elapsed_ms is not None:
                    state.metrics["tool_latencies_ms"].append(res.elapsed_ms)
                yield StructuredAgentEvent(
                    "tool_result",
                    state.step,
                    ToolResultBlock(res.tool_call_id, res.content, res.status),
                )
                for spec_event in self._emit_speculation(
                    tu.name, res.status, state.step
                ):
                    yield spec_event
            state.messages.append(
                {"role": "user", "content": [tool_result_message(r) for r in results]}
            )

            if _is_productive(uses, results, active_tool_map):
                state.consecutive_idle_steps = 0
            elif state.mode == "act":
                state.consecutive_idle_steps += 1
            if state.consecutive_idle_steps >= 4:
                raise RuntimeError(
                    "Watchdog triggered: 4 consecutive steps without productive tool calls."
                )

            while self.followup_queue:
                state.messages.append(to_dict(self.followup_queue.pop(0)))

        yield _final_event(
            self.config.max_steps,
            StructuredAgentResult(
                answer="step limit reached",
                messages=state.messages,
                steps=self.config.max_steps,
                tool_calls=state.tool_calls,
                stopped_by_limit=True,
                metrics=finalize_metrics(state.metrics),
            ),
        )

    # ── 内层循环 ──

    async def _run_inner_loop(
        self,
        state: RunState,
        active_registry: tuple[ToolSpec, ...],
        out_events: list[StructuredAgentEvent],
    ) -> list[dict[str, Any]] | None:
        """内层循环：compact → model call → retry/max_tokens。返回 accumulated 或 None（提前退出）。"""
        accumulated: list[dict[str, Any]] = []
        while True:
            if self._should_compact(state.messages):
                self._emit_hook(
                    HookRecord("on_compact", metadata={"messages": len(state.messages)})
                )
                assert self.compactor is not None
                state.messages = self.compactor(state.messages)
            state.messages = budget_messages_for_provider(state.messages)
            state.metrics["llm_calls"] += 1
            state.metrics["estimated_prompt_tokens"] += estimate_message_tokens(
                state.messages
            )
            started = perf_counter()

            (
                blocks,
                stop_reason,
                events,
                reasoning,
                new_err,
                switched,
            ) = await call_model_streaming(
                self.provider,
                self.fallback_provider,
                state.messages,
                active_registry,
                state.consecutive_errors,
                state.step,
            )
            state.consecutive_errors = new_err
            if switched:
                self.provider = self.fallback_provider  # 永久切换
                self._consecutive_errors = 0
            state.metrics["model_latencies_ms"].append(elapsed_ms(started))
            out_events.extend(events)
            out_events.append(StructuredAgentEvent("assistant", state.step, blocks))
            accumulated.extend(blocks)
            state.messages.append(
                to_dict(
                    AssistantMessage(
                        content=blocks_to_typed(blocks),
                        reasoning_content=reasoning_for_assistant(blocks, reasoning),
                    )
                )
            )

            if stop_reason == "error":
                retries = state.step_retries + 1
                state.step_retries = retries
                if retries <= 3:
                    await asyncio.sleep(0.5 * (2 ** (retries - 1)))
                    continue
                if not accumulated:
                    out_events.append(
                        _final_event(
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
                    )
                    return None
                return accumulated

            if stop_reason == "max_tokens":
                inc = estimate_text_tokens(
                    json.dumps(blocks, ensure_ascii=False, default=str)
                )
                state.consecutive_continuations = (
                    state.consecutive_continuations + 1 if inc < 500 else 0
                )
                if state.consecutive_continuations >= 3:
                    raise RuntimeError(
                        "Diminishing Returns: consecutive output token increments below 500 limit."
                    )
                state.messages.append(to_dict(UserMessage(content="continue")))
                continue

            state.consecutive_continuations = 0
            return accumulated

    # ── 辅助方法 ──

    def _emit_hook(self, record: HookRecord) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(record)

    def _should_compact(self, messages: list[dict[str, Any]]) -> bool:
        if self.compactor is None:
            return False
        if self.manual_compact_requested and self.manual_compact_requested():
            return True
        return (
            self.config.compact_threshold > 0
            and len(messages) > self.config.compact_threshold
        ) or (
            self.config.compact_token_threshold > 0
            and estimate_message_tokens(messages) > self.config.compact_token_threshold
        )

    def _initial_messages(
        self, question: str, mode: ExecutionMode = "act"
    ) -> list[dict[str, Any]]:
        typed: list[AgentMessage] = []
        notice = mode_notice(mode)
        if self.runtime_context_provider is not None:
            parts = self.runtime_context_provider(question)
            if notice:
                parts.append(notice)
            if parts:
                typed.append(SystemMessage(content="\n\n".join(p for p in parts if p)))
        elif notice:
            typed.append(SystemMessage(content=notice))
        typed.append(UserMessage(content=question))
        return convert_to_llm(typed)

    def _emit_speculation(
        self, tool_name: str | None, status: str, step: int
    ) -> Generator[StructuredAgentEvent, None, None]:
        if self.speculation_planner is None:
            return
        event = self.speculation_planner.plan(tool_name, status)
        if event is not None:
            yield StructuredAgentEvent("speculation", step, event)


# ── 模块级辅助 ──


def _resolve_permission_policy(
    project_root: Path | None, base: PermissionPolicy | None
) -> PermissionPolicy | None:
    if project_root is None:
        return base
    local = project_root / ".local" / "settings.json"
    root = project_root / "settings.json"
    settings_path = local if local.exists() else (root if root.exists() else None)
    if settings_path is None:
        return base
    from ..observability.permissions import (
        SettingsSandboxPermissionPolicy,
        CompositePermissionPolicy,
    )

    sandbox = SettingsSandboxPermissionPolicy(settings_path)
    return cast(PermissionPolicy, CompositePermissionPolicy(sandbox, base))


def _is_productive(
    uses: list[ToolUseBlock], results: list[ToolResult], tool_map: dict[str, ToolSpec]
) -> bool:
    for tu, res in zip(uses, results, strict=True):
        if res.status == "ok":
            spec = tool_map.get(tu.name)
            if spec and spec.read_only:
                return True
            if tu.name in (
                "write_file",
                "edit_file",
                "write_to_file",
                "replace_file_content",
                "multi_replace_file_content",
                "bash",
            ):
                return True
    return False


def _final_event(step: int, result: StructuredAgentResult) -> StructuredAgentEvent:
    return StructuredAgentEvent("final", step, result)
