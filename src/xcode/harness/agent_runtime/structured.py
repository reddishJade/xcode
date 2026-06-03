"""结构化工具调用执行框架。

本模块把模型响应看作 content blocks：text、tool_use、tool_result，并在
事件流中暴露模型输出、工具调用、工具结果和最终答案。

StructuredAgent 是 harness 层对 agent 核心循环的适配：将 Xcode 特定的
ToolSpec、权限、审计、压缩等配置映射为 AgentLoopConfig，委托给
agent/agent_loop.py 的 run_agent_loop 执行。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Generator, Iterator, cast

from ...agent.agent_loop import run_agent_loop
from ...agent.messages import convert_to_llm
from ...agent.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopResult,
    AgentMessage,
    AssistantMessage,
    MessageEndEvent,
    MessageUpdateEvent,
    SystemMessage,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
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
from ..skills import ApprovalCallback, ToolSpec

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
        self.cancellation_token.reset()

        # 构建初始消息
        initial_messages = self._initial_messages(question, effective_mode)

        # 适配 ToolSpec → AgentTool
        adapted_tools: list[Any] = adapt_tool_specs(
            active_registry,
            approval_callback=self.approval_callback,
            permission_policy=self.permission_policy,
        )

        # 构建 AgentLoopConfig
        loop_config = self._build_loop_config(effective_mode)

        # 构建 AgentContext
        context = AgentContext(
            messages=[],
            tools=adapted_tools,
        )

        # 事件翻译：收集事件并 yield
        translated_events: list[StructuredAgentEvent] = []
        step_counter = [0]

        def translate_emit(event: AgentEvent) -> None:
            translated = _translate_event(event, step_counter)
            if translated is not None:
                translated_events.extend(
                    translated if isinstance(translated, list) else [translated]
                )

        # 委托给 agent 核心循环
        result = await run_agent_loop(
            prompts=initial_messages,
            context=context,
            config=loop_config,
            emit=translate_emit,
            signal=self.cancellation_token,  # type: ignore[arg-type]
        )

        # 同步 provider 状态（wrapper 可能已切换到 fallback）
        if isinstance(self.provider, _FallbackSwitchingProvider):
            wrapper = self.provider
            if wrapper._using_fallback:
                self._original_provider = wrapper._fallback
            else:
                self._original_provider = wrapper._primary

        # yield 所有翻译后的事件
        for event in translated_events:
            yield event

        # 构建最终结果
        final = _build_structured_result(result, self.config.max_steps)
        yield _final_event(result.steps, final)

    # ── 配置构建 ──

    def _build_loop_config(self, mode: ExecutionMode) -> AgentLoopConfig:
        """将 harness 配置映射为 AgentLoopConfig。"""

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
            tool_results: list[Any],
        ) -> bool:
            # plan 模式不做空闲检测（阅读即探索，不算空转）
            if mode == "plan":
                return True
            return _is_productive_from_results(tool_calls, tool_results)

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
            get_steering_messages=self._drain_steer_queue,
            get_follow_up_messages=self._drain_followup_queue,
        )

    def _drain_steer_queue(self) -> list[AgentMessage]:
        msgs = list(self.steer_queue)
        self.steer_queue.clear()
        return msgs

    def _drain_followup_queue(self) -> list[AgentMessage]:
        msgs = list(self.followup_queue)
        self.followup_queue.clear()
        return msgs

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
    event: AgentEvent, step_counter: list[int]
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
            blocks = _assistant_to_raw_blocks(msg)
            return StructuredAgentEvent("assistant", step_counter[0], blocks)
        return None

    if isinstance(event, MessageEndEvent):
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
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments or {},
            })
    return blocks


def _build_structured_result(
    result: AgentLoopResult, max_steps: int
) -> StructuredAgentResult:
    """将 AgentLoopResult 转换为 StructuredAgentResult。"""
    # 提取文本答案
    answer = ""
    for msg in result.messages:
        if isinstance(msg, AssistantMessage):
            answer = text_from_blocks(
                [{"type": "text", "text": b.text} if isinstance(b, TextContent) else {}
                 for b in msg.content]
            )

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


def _is_productive_from_results(
    tool_calls: list[ToolCallContent], tool_results: list[Any]
) -> bool:
    """从工具调用和结果判断是否有生产力。

    与 _is_productive 逻辑一致：只有写操作（write_file、edit_file、bash 等）
    才算有生产力，read_file/glob/grep 等只读操作不算。
    """
    _write_tools = {
        "write_file", "edit_file", "write_to_file",
        "replace_file_content", "multi_replace_file_content", "bash",
    }
    for tc, res in zip(tool_calls, tool_results, strict=False):
        is_ok = (
            (hasattr(res, "is_error") and not res.is_error)
            or (hasattr(res, "status") and res.status == "ok")
        )
        if is_ok and tc.name in _write_tools:
            return True
    return False


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
        if hasattr(res, "status") and res.status == "ok":
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
