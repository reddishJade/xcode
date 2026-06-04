"""结构化工具调用执行框架。

本模块把模型响应看作 content blocks：text、tool_use、tool_result，并在
事件流中暴露模型输出、工具调用、工具结果和最终答案。

StructuredAgent 是 harness 层对 agent/Agent 的适配：将 Xcode 特定的
ToolSpec、权限、审计、压缩等配置映射为 AgentLoopConfig，委托给
agent/Agent.run() 执行。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Generator, Iterator, cast

from ...agent.agent import Agent
from ...agent.messages import convert_to_llm
from ...agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopResult,
    AgentMessage,
    AssistantMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageUpdateEvent,
    SystemMessage,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessage,
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
from .execution_modes import mode_notice, policy_for_mode
from .tool_adapter import adapt_tool_specs
from ..config import AgentConfig, ExecutionMode
from ..observability import AuditRecord, HookManager, HookRecord, PermissionPolicy
from ..skills import ApprovalCallback, ToolSpec, stringify_tool_input

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

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
    ) -> AsyncIterator[ProviderEvent]:
        provider = self._fallback if self._using_fallback else self._primary
        try:
            async for event in provider.stream(messages, tools):
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
                async for event in self._fallback.stream(messages, tools):
                    yield event
            else:
                raise


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
        speculation_planner: Any = None,
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

        # 适配 ToolSpec → AgentTool，创建 Agent 实例
        adapted_tools: list[Any] = adapt_tool_specs(
            registry,
            approval_callback=approval_callback,
            permission_policy=self.permission_policy,
        )
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

        # 实时流式：消费 Agent.run_stream()，边跑边翻译边 yield
        step_counter = [0]
        text_seen: dict[int, str] = {}

        async for event in self._agent.run_stream(
            initial_messages,
            loop_config,
            signal=self.cancellation_token,  # type: ignore[arg-type]
        ):
            translated = _translate_event(event, step_counter, text_seen)
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
                            event.tool_name, status, step_counter[0]
                        ):
                            yield spec_event

        # 循环结束，从 Agent 获取结果
        result = self._agent._last_result
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

    # ── 配置构建 ──

    def _build_loop_config(self, mode: ExecutionMode) -> AgentLoopConfig:
        """将 harness 配置映射为 AgentLoopConfig。

        队列 drain（get_steering_messages / get_follow_up_messages）
        由 Agent 层注入，此处不设置。
        """
        policy = policy_for_mode(mode)

        def should_compact(messages: list[AgentMessage]) -> bool:
            return self._should_compact([to_dict(m) for m in messages])

        def compact(messages: list[AgentMessage]) -> list[AgentMessage]:
            self._emit_hook(
                HookRecord("on_compact", metadata={"messages": len(messages)})
            )
            if self.compactor is None:
                return messages
            dict_messages = [to_dict(m) for m in messages]
            self.compactor(dict_messages)
            # compactor 通过 dict 操作，返回值由 harness 层管理
            return messages

        def is_tool_productive(
            tool_calls: list[ToolCallContent],
            tool_results: list[ToolResultMessage],
        ) -> bool:
            if mode == "plan":
                return True
            return _is_productive(
                [
                    ToolUseBlock(id="", name=tc.name, input=tc.arguments or {})
                    for tc in tool_calls
                ],
                tool_results,
                self.tool_map,
            )

        def before_tool(
            ctx: BeforeToolCallContext, signal: Any
        ) -> BeforeToolCallResult | None:
            tc = ctx.tool_call
            args = ctx.args
            action_input = stringify_tool_input(args)

            # 执行模式策略检查
            decision = policy.check_call(
                ToolUseBlock(id=tc.id, name=tc.name, input=args)
            )
            if decision == "deny":
                return BeforeToolCallResult(
                    block=True, reason=f"permission denied for tool: {tc.name}"
                )
            if decision == "ask":
                if self.approval_callback is None or self.tool_map.get(tc.name) is None:
                    return BeforeToolCallResult(
                        block=True, reason=f"tool requires approval: {tc.name}"
                    )
                hitl = self.approval_callback(self.tool_map[tc.name], args)
                if hitl.decision == "deny":
                    return BeforeToolCallResult(
                        block=True, reason=f"tool {tc.name} denied by user"
                    )

            # pre_tool 钩子
            if self.hook_manager is not None:
                self.hook_manager.emit(
                    HookRecord(
                        "pre_tool", tool=tc.name, input=action_input
                    )
                )
            return None

        def after_tool(
            ctx: AfterToolCallContext, signal: Any
        ) -> AfterToolCallResult | None:
            tc = ctx.tool_call
            args = ctx.args
            action_input = stringify_tool_input(args)

            result_content_text = ""
            if ctx.result and ctx.result.content:
                result_content_text = "".join(
                    c.text for c in ctx.result.content if isinstance(c, TextContent)
                )

            if ctx.is_error:
                if self.hook_manager is not None:
                    self.hook_manager.emit(
                        HookRecord(
                            "on_error",
                            tool=tc.name,
                            input=action_input,
                            error=result_content_text,
                        )
                    )
            else:
                if self.hook_manager is not None:
                    self.hook_manager.emit(
                        HookRecord(
                            "post_tool",
                            tool=tc.name,
                            input=action_input,
                            output=result_content_text,
                        )
                    )

            if self.audit_logger is not None:
                from ..observability.audit import redact_text

                self.audit_logger(
                    AuditRecord(
                        session_id=self.session_id,
                        tool=tc.name,
                        static_risk=self.tool_map.get(
                            tc.name, ToolSpec("", "", "", lambda _: "")
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
            return None

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
            should_compact=should_compact if self.compactor else None,
            compact=compact if self.compactor else None,
            is_tool_productive=is_tool_productive,
            before_tool_call=before_tool,
            after_tool_call=after_tool,
        )

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
    ) -> list[AgentMessage]:
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
    step_counter: list[int],
    _text_seen: dict[int, str] | None = None,
) -> StructuredAgentEvent | list[StructuredAgentEvent] | None:
    """将 AgentEvent 翻译为 StructuredAgentEvent。"""
    from ...agent.types import AgentStartEvent, TurnStartEvent

    if isinstance(event, AgentStartEvent):
        return None

    if isinstance(event, TurnStartEvent):
        step_counter[0] += 1
        return None

    if isinstance(event, MessageUpdateEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    step = step_counter[0]
                    prev = _text_seen.get(step, "") if _text_seen is not None else ""
                    full = block.text
                    delta = full[len(prev):]
                    if not delta:
                        return None
                    if _text_seen is not None:
                        _text_seen[step] = full
                    return StructuredAgentEvent("text_delta", step, delta)
        return None

    if isinstance(event, MessageEndEvent):
        msg = event.message
        if isinstance(msg, AssistantMessage) and msg.content:
            blocks = _assistant_to_raw_blocks(msg)
            if blocks:
                return StructuredAgentEvent("assistant", step_counter[0], blocks)
        return None

    if isinstance(event, ToolExecutionStartEvent):
        tu = ToolUseBlock(
            id=event.tool_call_id,
            name=event.tool_name,
            input=event.args or {},
        )
        return StructuredAgentEvent("tool_use", step_counter[0], tu)

    if isinstance(event, ToolExecutionEndEvent):
        return StructuredAgentEvent(
            "tool_result",
            step_counter[0],
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
    # 提取文本答案（拼接所有 AssistantMessage 中的文本）
    answer_parts: list[str] = []
    for msg in result.messages:
        if isinstance(msg, AssistantMessage):
            extracted = text_from_blocks(
                [
                    {"type": "text", "text": b.text}
                    if isinstance(b, TextContent)
                    else {}
                    for b in msg.content
                ]
            )
            if extracted:
                answer_parts.append(extracted)
    answer = " ".join(answer_parts)

    # 提取工具调用
    tool_calls: list[ToolUseBlock] = []
    for msg in result.messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCallContent):
                    tool_calls.append(
                        ToolUseBlock(
                            id=block.id,
                            name=block.name,
                            input=block.arguments or {},
                        )
                    )

    # 转换消息为 dict
    messages = [to_dict(m) for m in result.messages]

    # 构建指标
    metrics = None
    if result.metrics:
        metrics = {
            "llm_calls": result.metrics.llm_calls,
            "tool_calls": result.metrics.tool_calls,
            "estimated_prompt_tokens": 0,
            "model_latencies_ms": result.metrics.model_latencies_ms,
            "tool_latencies_ms": result.metrics.tool_latencies_ms,
        }

    # 当 agent 被看门狗或步骤限制停止时，追加原因
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
    return cast(PermissionPolicy, CompositePermissionPolicy(sandbox, base))


def _is_productive(
    uses: list[ToolUseBlock], results: list[Any], tool_map: dict[str, ToolSpec]
) -> bool:
    for tu, res in zip(uses, results, strict=True):
        is_ok = (hasattr(res, "is_error") and not res.is_error) or (
            hasattr(res, "status") and res.status == "ok"
        )
        if not is_ok:
            continue
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
