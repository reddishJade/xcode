"""结构化工具调用执行框架。

本模块把模型响应看作 content blocks：text、tool_use、tool_result，并在
事件流中暴露模型输出、工具调用、工具结果和最终答案。

StructuredAgent 是 harness 层对 agent/Agent 的适配：将 Xcode 特定的
ToolSpec、权限、审计、压缩等配置映射为 AgentLoopConfig，委托给
agent/Agent.run() 执行。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ...agent.agent import Agent
from ...agent.messages import convert_to_llm
from ...agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentStartEvent,
    AssistantMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SystemMessage,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ThinkingUpdateEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)
from .agent_helpers import (
    run_coro_sync,
    aiter_to_sync_iter,
    text_from_blocks,
    to_dict,
)
from .cancellation import CancellationToken
from .compaction import CompactController, estimate_message_tokens
from xcode.ai.events import ProviderEvent, ToolCall as ToolUseBlock
from xcode.ai.providers.protocol import ModelProvider
from xcode.ai.types import StreamOptions, ToolDefinition
from .execution_modes import ActPolicy, mode_notice, PlanPolicy, policy_for_mode
from .tool_adapter import adapt_tool_specs
from ..config import AgentConfig, ExecutionMode
from ..observability import (
    AuditRecord,
    HookManager,
    HookRecord,
    PermissionPolicy,
    redact_text,
)
from ..skills import ApprovalCallback, ToolSpec, stringify_tool_input

if TYPE_CHECKING:
    from xcode.experimental.speculation import SpeculationPlanner

__all__ = ["StructuredAgent", "StructuredAgentEvent", "StructuredAgentResult"]

StructuredCompactor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
RuntimeContextProvider = Callable[[str], list[str]]


class _FallbackSwitchingProvider:
    """Provider wrapper that switches to fallback after consecutive errors.

    Tracks consecutive error count across calls and switches to fallback_provider
    when the threshold is reached. Used by harness layer for model fallback.
    """

    def __init__(
        self,
        primary: ModelProvider,
        fallback: ModelProvider,
        error_threshold: int = 3,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._error_threshold = error_threshold
        self._consecutive_errors: int = 0
        self._using_fallback: bool = False

    @property
    def active_provider(self) -> ModelProvider:
        return self._fallback if self._using_fallback else self._primary

    @property
    def model(self) -> str:
        return getattr(self.active_provider, "model", "unknown")

    @property
    def thinking(self) -> bool:
        return getattr(self.active_provider, "thinking", True)

    @property
    def reasoning_effort(self) -> str | None:
        return getattr(self.active_provider, "reasoning_effort", None)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        provider = self._fallback if self._using_fallback else self._primary
        try:
            async for event in self._stream_with(
                provider, messages, tools, options, kwargs
            ):
                self._consecutive_errors = 0
                yield event
        except Exception:
            self._consecutive_errors += 1
            if (
                not self._using_fallback
                and self._consecutive_errors >= self._error_threshold
                and self._fallback is not None
            ):
                self._using_fallback = True
                async for event in self._stream_with(
                    self._fallback, messages, tools, options, kwargs
                ):
                    yield event
            else:
                raise

    @staticmethod
    async def _stream_with(
        provider: Any,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        options: StreamOptions | None,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[ProviderEvent]:
        try:
            async for event in provider.stream(
                messages, tools, options=options, **kwargs
            ):
                yield event
        except TypeError:
            async for event in provider.stream(messages, tools):
                yield event


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


@dataclass
class _StreamTranslationState:
    step: int = 0
    text_seen: dict[int, str] = field(default_factory=dict)


class StructuredAgent:
    """与 provider 解耦的结构化工具调用循环。

    harness 层适配器：将 Xcode 特定配置映射为 AgentLoopConfig，
    委托 agent 核心循环执行，通过事件翻译保持 StructuredAgentEvent
    接口不变。
    """

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
        speculation_planner: SpeculationPlanner | None = None,
        fallback_provider: ModelProvider | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.provider: ModelProvider = provider
        if fallback_provider is not None:
            self.provider = _FallbackSwitchingProvider(provider, fallback_provider)
        self._original_provider = provider
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
        self._current_mode: ExecutionMode = "act"
        self._progress_steps_without_update: int = 0
        self._last_progress_step: int = 0

        # 适配 ToolSpec → AgentTool，创建 Agent 实例
        adapted_tools: list[Any] = adapt_tool_specs(
            registry,
            approval_callback=approval_callback,
            permission_policy=self.permission_policy,
        )
        adapted_tools.extend(self._build_mode_switch_tools())
        self._agent = Agent(adapted_tools)

    # ── 公共 API ──

    def steer(self, msg: AgentMessage) -> None:
        self._agent.steer(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        self._agent.follow_up(msg)

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
        self.cancellation_token.reset()

        # 构建初始消息
        initial_messages = self._initial_messages(question, effective_mode)

        # 重新适配工具（执行模式可能过滤了工具集）
        adapted_tools: list[Any] = adapt_tool_specs(
            active_registry,
            approval_callback=self.approval_callback,
            permission_policy=self.permission_policy,
        )
        self._agent = Agent(adapted_tools)

        # 构建 AgentLoopConfig（含 harness 钩子）
        loop_config = self._build_loop_config(effective_mode)

        # 发射 before_agent_start 钩子
        self._emit_hook(
            HookRecord("before_agent_start", metadata={
                "question": question, "mode": effective_mode,
            })
        )

        # 实时流式：消费 Agent.run_stream()，边跑边翻译边 yield
        translation_state = _StreamTranslationState()

        async for event in self._agent.run_stream(
            initial_messages,
            loop_config,
            signal=self.cancellation_token,
        ):
            translated = _translate_event(event, translation_state)
            if translated is not None:
                for te in translated if isinstance(translated, list) else [translated]:
                    yield te
                    # 工具执行完成后发射推测事件
                    if (
                        isinstance(event, ToolExecutionEndEvent)
                        and te.type == "tool_result"
                    ):
                        status = te.data.status if hasattr(te.data, "status") else "ok"
                        for spec_event in self._emit_speculation(
                            event.tool_name, status, translation_state.step
                        ):
                            yield spec_event

        # 循环结束，从 Agent 获取结果
        result = self._agent.last_result
        assert result is not None

        # 同步 provider 状态（wrapper 可能已切换到 fallback）
        if isinstance(self.provider, _FallbackSwitchingProvider):
            wrapper = self.provider
            if wrapper._using_fallback:
                self._original_provider = wrapper._fallback
            else:
                self._original_provider = wrapper._primary

        # 构建最终结果
        final = _build_structured_result(result, self.config.max_steps)
        yield _final_event(result.steps, final)

    # ── 模式切换 ──

    def _build_mode_switch_tools(self) -> list[Any]:
        from .tool_adapter import adapt_tool_specs

        plan_tool = ToolSpec(
            name="enter_plan_mode",
            description="Switch to Plan Mode: read-only tools only. Call this before making changes to investigate first.",
            input_hint="empty",
            handler=lambda _input: self._switch_to_plan(),
            risk="low",
        )
        act_tool = ToolSpec(
            name="exit_plan_mode",
            description="Exit Plan Mode: return to full tool access. Call this with a concise summary of your plan.",
            input_hint="plan_summary",
            handler=lambda _input: self._switch_to_act(
                _input.get("plan_summary", "") if isinstance(_input, dict) else str(_input)
            ),
            risk="low",
        )
        return adapt_tool_specs((plan_tool, act_tool))

    def _switch_to_plan(self) -> str:
        self._current_mode = "plan"
        return "Entered Plan Mode. Tools are limited to read-only. Investigate and report a plan."

    def _switch_to_act(self, plan_summary: str | None = None) -> str:
        self._current_mode = "act"
        if plan_summary:
            self.steer(SystemMessage(content=(
                f"<plan-summary>\n{plan_summary}\n</plan-summary>\n"
                "The plan above was prepared during Plan Mode. Execute it now."
            )))
        self._agent._tools = self._tools_for_mode(self.registry, "act")
        return "Exited Plan Mode. Full tool access restored."

    def _tools_for_mode(
        self, registry: tuple[ToolSpec, ...], mode: ExecutionMode
    ) -> list[Any]:
        policy = policy_for_mode(mode)
        filtered = policy.filter_tools(registry)
        adapted = adapt_tool_specs(
            filtered,
            approval_callback=self.approval_callback,
            permission_policy=self.permission_policy,
        )
        adapted.extend(self._build_mode_switch_tools())
        return adapted

    # ── 配置构建 ──

    def _build_loop_config(self, mode: ExecutionMode) -> AgentLoopConfig:
        """将 harness 配置映射为 AgentLoopConfig。

        队列 drain（get_steering_messages / get_follow_up_messages）
        由 Agent 层注入，此处不设置。
        """
        should_compact = self._loop_should_compact if self.compactor else None
        compact = self._loop_compact if self.compactor else None

        return AgentLoopConfig(
            provider=self.provider,
            convert_to_llm=convert_to_llm,
            max_steps=self.config.max_steps,
            max_step_retries=3,
            retry_backoff_base=0.5,
            max_tokens_continuation=True,
            max_consecutive_continuations=3,
            min_continuation_tokens=500,
            watchdog_repeated_tool_limit=self.config.watchdog_repeated_tool_limit,
            max_consecutive_idle_steps=4,
            should_compact=should_compact,
            compact=compact,
            is_tool_productive=self._loop_is_tool_productive(mode),
            before_tool_call=self._loop_before_tool(mode),
            after_tool_call=self._loop_after_tool,
            prepare_next_turn=self._loop_prepare_next_turn,
        )

    # ── 辅助方法 ──

    def _loop_should_compact(self, messages: list[AgentMessage]) -> bool:
        return self._should_compact([to_dict(m) for m in messages])

    def _loop_compact(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        self._emit_hook(HookRecord("on_compact", metadata={"messages": len(messages)}))
        if self.compactor is None:
            return messages
        dict_messages = [to_dict(m) for m in messages]
        self.compactor(dict_messages)
        # compactor 通过 dict 操作，返回值由 harness 层管理。
        return messages

    def _loop_prepare_next_turn(
        self,
    ) -> AgentLoopTurnUpdate | None:
        self._progress_steps_without_update += 1
        if self._progress_steps_without_update >= 5:
            self._progress_steps_without_update = 0
            self.steer(UserMessage(
                content="<reminder>You have gone several turns without updating task progress. "
                        "Use update_task or save_task_progress to record progress before continuing.</reminder>"
            ))
        return None

    def _loop_is_tool_productive(
        self, mode: ExecutionMode
    ) -> Callable[[list[ToolCallContent], list[ToolResultMessage]], bool]:
        def is_tool_productive(
            tool_calls: list[ToolCallContent],
            tool_results: list[ToolResultMessage],
        ) -> bool:
            if self._current_mode == "plan":
                return True
            return _tool_results_count_as_progress(
                [
                    ToolUseBlock(
                        id="",
                        name=tool_call.name,
                        input=tool_call.arguments or {},
                    )
                    for tool_call in tool_calls
                ],
                tool_results,
                self.tool_map,
            )

        return is_tool_productive

    def _loop_before_tool(
        self, mode: ExecutionMode
    ) -> Callable[[BeforeToolCallContext, Any], BeforeToolCallResult | None]:
        def before_tool(
            ctx: BeforeToolCallContext, _signal: Any
        ) -> BeforeToolCallResult | None:
            tool_call = ctx.tool_call
            args = ctx.args
            action_input = stringify_tool_input(args)

            effective_policy = policy_for_mode(self._current_mode)
            decision = effective_policy.check_call(
                ToolUseBlock(id=tool_call.id, name=tool_call.name, input=args)
            )
            if decision == "deny":
                return BeforeToolCallResult(
                    block=True,
                    reason=f"tool not allowed in {self._current_mode} mode: {tool_call.name}",
                )
            if decision == "ask":
                approval = self._request_tool_approval(tool_call.name, args)
                if approval is not None:
                    return approval

            self._emit_hook(
                HookRecord("pre_tool", tool=tool_call.name, input=action_input)
            )
            return None

        return before_tool

    def _request_tool_approval(
        self, tool_name: str, args: dict[str, Any]
    ) -> BeforeToolCallResult | None:
        if self.approval_callback is None or self.tool_map.get(tool_name) is None:
            return BeforeToolCallResult(
                block=True,
                reason=f"tool requires approval: {tool_name}",
            )
        hitl = self.approval_callback(self.tool_map[tool_name], args)
        if hitl.decision == "deny":
            return BeforeToolCallResult(
                block=True,
                reason=f"tool {tool_name} denied by user",
            )
        return None

    PROGRESS_TOOL_NAMES = frozenset({
        "save_task_progress", "resume_task_progress",
        "update_task", "create_task",
    })

    def _loop_after_tool(
        self, ctx: AfterToolCallContext, _signal: Any
    ) -> AfterToolCallResult | None:
        if ctx.tool_call.name in self.PROGRESS_TOOL_NAMES:
            self._progress_steps_without_update = 0
        action_input = stringify_tool_input(ctx.args)
        result_content_text = _tool_result_text(ctx)

        self._emit_tool_hook(ctx, action_input, result_content_text)
        self._emit_audit_record(ctx, action_input, result_content_text)
        return None

    def _emit_tool_hook(
        self,
        ctx: AfterToolCallContext,
        action_input: str,
        result_content_text: str,
    ) -> None:
        tool_call = ctx.tool_call
        if ctx.is_error:
            self._emit_hook(
                HookRecord(
                    "on_error",
                    tool=tool_call.name,
                    input=action_input,
                    error=result_content_text,
                )
            )
            return
        self._emit_hook(
            HookRecord(
                "post_tool",
                tool=tool_call.name,
                input=action_input,
                output=result_content_text,
            )
        )

    def _emit_audit_record(
        self,
        ctx: AfterToolCallContext,
        action_input: str,
        result_content_text: str,
    ) -> None:
        if self.audit_logger is None:
            return
        tool_call = ctx.tool_call
        self.audit_logger(
            AuditRecord(
                session_id=self.session_id,
                tool=tool_call.name,
                static_risk=self.tool_map.get(
                    tool_call.name, ToolSpec("", "", "", lambda _: "")
                ).risk
                or "low",
                dynamic_decision="allow",
                policy_decision=None,
                final_status="error" if ctx.is_error else "ok",
                approved=True,
                redacted_input=redact_text(action_input),
                redacted_output=redact_text(result_content_text),
            )
        )

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
    ) -> list[AgentMessage]:
        self._current_mode = mode
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
        return typed

    def _emit_speculation(
        self, tool_name: str | None, status: str, step: int
    ) -> Generator[StructuredAgentEvent, None, None]:
        if self.speculation_planner is None:
            return
        event = self.speculation_planner.plan(tool_name, status)
        if event is not None:
            yield StructuredAgentEvent("speculation", step, event)


# ── 事件翻译 ──


def _translate_event(
    event: AgentEvent,
    state: _StreamTranslationState,
) -> StructuredAgentEvent | list[StructuredAgentEvent] | None:
    """将 AgentEvent 翻译为 StructuredAgentEvent。"""

    if isinstance(event, AgentStartEvent):
        return None

    if isinstance(event, TurnStartEvent):
        state.step += 1
        return None

    if isinstance(event, MessageUpdateEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    step = state.step
                    prev = state.text_seen.get(step, "")
                    full = block.text
                    delta = full[len(prev) :]
                    if not delta:
                        return None
                    state.text_seen[step] = full
                    return StructuredAgentEvent("text_delta", step, delta)
        return None

    if isinstance(event, MessageStartEvent):
        return StructuredAgentEvent("message_start", state.step, event.message)

    if isinstance(event, TurnEndEvent):
        return StructuredAgentEvent(
            "turn_end",
            state.step,
            {
                "tool_results": [
                    {"tool_call_id": r.tool_call_id, "content": str(r.content)}
                    for r in event.tool_results
                ]
            },
        )

    if isinstance(event, ThinkingUpdateEvent):
        return StructuredAgentEvent(
            "reasoning_delta", state.step, event.reasoning_content
        )

    if isinstance(event, MessageEndEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            blocks = _assistant_to_raw_blocks(msg)
            if blocks:
                return StructuredAgentEvent("assistant", state.step, blocks)
        return None

    if isinstance(event, ToolExecutionStartEvent):
        tool_use = ToolUseBlock(
            id=event.tool_call_id,
            name=event.tool_name,
            input=event.args or {},
        )
        return StructuredAgentEvent("tool_use", state.step, tool_use)

    if isinstance(event, ToolExecutionUpdateEvent):
        return StructuredAgentEvent(
            "tool_update",
            state.step,
            {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "partial_result": str(event.partial_result)
                if event.partial_result
                else "",
            },
        )

    if isinstance(event, ToolExecutionEndEvent):
        return StructuredAgentEvent(
            "tool_result",
            state.step,
            ToolResultBlock(
                tool_use_id=event.tool_call_id,
                content=str(event.result.content) if event.result else "",
                status="error" if event.is_error else "ok",
            ),
        )

    return None


def _assistant_to_raw_blocks(msg: AssistantMessage) -> list[dict[str, Any]]:
    """将 AssistantMessage 转换为 raw block 列表。"""
    blocks: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCallContent):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments or {},
                }
            )
    return blocks


def _build_structured_result(
    result: AgentLoopResult, max_steps: int
) -> StructuredAgentResult:
    """将 AgentLoopResult 转换为 StructuredAgentResult。"""
    answer_parts: list[str] = []
    tool_calls: list[ToolUseBlock] = []
    messages: list[dict[str, Any]] = []
    for msg in result.messages:
        messages.append(to_dict(msg))
        if not isinstance(msg, AssistantMessage):
            continue
        extracted = text_from_blocks(
            [
                {"type": "text", "text": b.text} if isinstance(b, TextContent) else {}
                for b in msg.content
            ]
        )
        if extracted:
            answer_parts.append(extracted)
        for block in msg.content:
            if isinstance(block, ToolCallContent):
                tool_calls.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.arguments or {},
                    )
                )

    answer = " ".join(answer_parts)
    metrics = None
    if result.metrics:
        metrics = {
            "llm_calls": result.metrics.llm_calls,
            "tool_calls": result.metrics.tool_calls,
            "estimated_prompt_tokens": 0,
            "model_latencies_ms": result.metrics.model_latencies_ms,
            "tool_latencies_ms": result.metrics.tool_latencies_ms,
        }

    if result.stopped_by_watchdog and result.watchdog_reason:
        if answer:
            answer = answer + " " + result.watchdog_reason
        else:
            answer = result.watchdog_reason
    elif result.stopped_by_limit and not answer:
        answer = "step limit reached"

    return StructuredAgentResult(
        answer=answer,
        messages=messages,
        steps=result.steps,
        tool_calls=tool_calls,
        stopped_by_limit=result.stopped_by_limit,
        metrics=metrics,
        stopped_by_watchdog=result.stopped_by_watchdog,
        watchdog_reason=result.watchdog_reason,
    )


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
    return CompositePermissionPolicy(sandbox, base)


def _tool_results_count_as_progress(
    tool_uses: list[ToolUseBlock],
    tool_results: list[Any],
    tool_map: dict[str, ToolSpec],
) -> bool:
    for tool_use, tool_result in zip(tool_uses, tool_results, strict=True):
        is_ok = (hasattr(tool_result, "is_error") and not tool_result.is_error) or (
            hasattr(tool_result, "status") and tool_result.status == "ok"
        )
        if not is_ok:
            continue
        spec = tool_map.get(tool_use.name)
        if spec and spec.counts_as_progress is not None:
            return spec.counts_as_progress
        if spec and spec.read_only:
            return True
    return False


def _tool_result_text(ctx: AfterToolCallContext) -> str:
    if not ctx.result or not ctx.result.content:
        return ""
    return "".join(c.text for c in ctx.result.content if isinstance(c, TextContent))


def _final_event(step: int, result: StructuredAgentResult) -> StructuredAgentEvent:
    return StructuredAgentEvent("final", step, result)
