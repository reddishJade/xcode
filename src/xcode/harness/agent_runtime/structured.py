"""StructuredAgent — harness 层对 agent/Agent 的适配。

将 Xcode 特定的 ToolSpec、权限、审计、压缩等配置映射为 AgentLoopConfig，
委托给 agent/Agent.run() 执行。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from ...agent.agent import Agent
from ...agent.messages import convert_to_llm
from ...agent.config import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from ...agent.protocols import ContentBlock
from xcode.ai.events import ToolCall as ToolUseBlock
from xcode.ai.providers.protocol import ModelProvider
from xcode.agent.types import TextContent, ToolCallContent
from .agent_helpers import (
    run_coro_sync,
    aiter_to_sync_iter,
    to_dict,
)
from .cancellation import CancellationToken
from .compaction import CompactController, estimate_message_tokens
from .event_translation import (
    _StreamTranslationState,
    _translate_event,
    StructuredAgentEvent,
)
from .execution_modes import mode_notice, policy_for_mode
from .fallback import _FallbackSwitchingProvider, _FallbackWithRetryPrimary
from .result import (
    _build_structured_result,
    _final_event,
    _tool_result_text,
    RunState,
    StructuredAgentResult,
)
from .session import AgentSession, InMemoryAgentSession
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


__all__ = ["StructuredAgent"]

StructuredCompactor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
RuntimeContextProvider = Callable[[str], list[str]]


@dataclass(frozen=True)
class TurnSnapshot:
    """单个 turn 使用的运行期快照。"""

    config: AgentConfig
    registry: tuple[ToolSpec, ...]
    tool_map: dict[str, ToolSpec]
    approval_callback: ApprovalCallback | None
    permission_policy: PermissionPolicy | None
    provider: ModelProvider
    runtime_context_provider: RuntimeContextProvider | None


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
        fallback_provider: ModelProvider | None = None,
        project_root: Path | None = None,
        session: AgentSession | None = None,
    ) -> None:
        self.provider: ModelProvider = provider
        if fallback_provider is not None:
            self.provider = _FallbackWithRetryPrimary(provider, fallback_provider)
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
        self._consecutive_errors: int = 0
        self._current_mode: ExecutionMode = "act"
        self._plan_enter_step: int = 0
        self._max_plan_turns: int = (
            8  # plan mode 最大探索深度：平衡探索全面性与防止过度推理
        )
        self._plan_pending_confirmation: bool = False
        self._progress_steps_without_update: int = 0
        self._last_progress_step: int = 0

        self._session = session or InMemoryAgentSession()

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

    def confirm_plan(self) -> None:
        """确认 plan，允许执行 write 工具。"""
        self._plan_pending_confirmation = False

    def clear_history(self) -> None:
        """清空会话历史。对应 /clear 命令。"""
        self._session.clear()

    def load_history(self, messages: list[AgentMessage]) -> None:
        """用外部会话记录替换当前会话历史。"""
        self._session.load(messages)

    def load_run_state(self, run_state: RunState) -> None:
        """用可序列化运行状态恢复当前会话。"""
        self._session.load(_messages_from_run_state(run_state))
        if run_state.current_mode in {"act", "plan", "review"}:
            self._current_mode = cast(ExecutionMode, run_state.current_mode)

    def history_messages(self) -> list[AgentMessage]:
        """返回当前会话历史副本。"""
        return self._session.messages()

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
        snapshot = self._turn_snapshot()
        effective_mode = mode or snapshot.config.execution_mode
        policy = policy_for_mode(effective_mode)
        active_registry = policy.filter_tools(snapshot.registry)
        self.cancellation_token.reset()

        # 构建 provider 上下文和本轮新增消息。
        context_messages = self._turn_context_messages(
            question, effective_mode, snapshot
        )
        history_messages = context_messages + self.history_messages()
        turn_messages: list[AgentMessage] = [UserMessage(content=question)]

        # 重新适配工具（执行模式可能过滤了工具集）
        adapted_tools: list[Any] = adapt_tool_specs(
            active_registry,
            approval_callback=snapshot.approval_callback,
            permission_policy=snapshot.permission_policy,
        )
        self._agent = Agent(adapted_tools)

        # 构建 AgentLoopConfig（含 harness 钩子）
        loop_config = self._build_loop_config(effective_mode, snapshot)

        # 发射 before_agent_start 钩子
        self._emit_hook(
            HookRecord(
                "before_agent_start",
                metadata={
                    "question": question,
                    "mode": effective_mode,
                },
            )
        )

        # 实时流式：消费 Agent.run_stream()，边跑边翻译边 yield
        translation_state = _StreamTranslationState()

        async for event in self._agent.run_stream(
            turn_messages,
            loop_config,
            signal=self.cancellation_token,
            history=history_messages,
        ):
            translated = _translate_event(event, translation_state)
            if translated is not None:
                for te in translated if isinstance(translated, list) else [translated]:
                    yield te

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

        # 保存本轮对话到历史
        self._save_turn_to_history(result.messages)

        # 构建最终结果
        visible_result = (
            replace(result, messages=context_messages + result.messages)
            if context_messages
            else result
        )
        final = _build_structured_result(
            visible_result, snapshot.config.max_steps, self._current_mode
        )
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
                _input.get("plan_summary", "")
                if isinstance(_input, dict)
                else str(_input)
            ),
            risk="low",
        )
        return adapt_tool_specs((plan_tool, act_tool))

    def _switch_to_plan(self) -> str:
        self._current_mode = "plan"
        self._plan_enter_step = 0
        return (
            "Entered Plan Mode. Tools are limited to read-only. "
            "Investigate and report a plan. "
            "Use exit_plan_mode with a concise plan summary to return to full tool access."
        )

    def _plan_confirmation_required(self, ctx: BeforeToolCallContext) -> bool:
        if not self._plan_pending_confirmation:
            return False
        policy = policy_for_mode("plan")
        decision = policy.check_call(
            ToolUseBlock(id=ctx.tool_call.id, name=ctx.tool_call.name, input=ctx.args)
        )
        return decision != "allow"

    def _switch_to_act(self, plan_summary: str | None = None) -> str:
        self._current_mode = "act"
        self._plan_pending_confirmation = True
        if plan_summary:
            self.steer(
                SystemMessage(
                    content=(
                        f"<plan-summary>\n{plan_summary}\n</plan-summary>\n"
                        "A plan has been prepared. Confirm with the user before executing write tools."
                    )
                )
            )
        self._agent.update_tools(self._tools_for_mode(self.registry, "act"))
        return "Plan ready. Present it to the user for confirmation."

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

    def _build_loop_config(
        self, mode: ExecutionMode, snapshot: TurnSnapshot
    ) -> AgentLoopConfig:
        """将 harness 配置映射为 AgentLoopConfig。

        队列 drain（get_steering_messages / get_follow_up_messages）
        由 Agent 层注入，此处不设置。
        """
        should_compact = self._loop_should_compact(snapshot) if self.compactor else None
        compact = self._loop_compact if self.compactor else None

        return AgentLoopConfig(
            provider=snapshot.provider,
            convert_to_llm=convert_to_llm,
            max_steps=snapshot.config.max_steps,
            max_step_retries=3,
            retry_backoff_base=0.5,
            max_tokens_continuation=True,
            max_consecutive_continuations=3,
            min_continuation_tokens=500,
            watchdog_repeated_tool_limit=snapshot.config.watchdog_repeated_tool_limit,
            max_consecutive_idle_steps=4,
            should_compact=should_compact,
            compact=compact,
            is_tool_productive=self._loop_is_tool_productive(mode, snapshot),
            before_tool_call=self._loop_before_tool(mode, snapshot),
            after_tool_call=self._loop_after_tool(snapshot),
            prepare_next_turn=self._loop_prepare_next_turn(snapshot),
        )

    # ── 辅助方法 ──

    def _loop_should_compact(
        self, snapshot: TurnSnapshot
    ) -> Callable[[list[AgentMessage]], bool]:
        def should_compact(messages: list[AgentMessage]) -> bool:
            return self._should_compact([to_dict(m) for m in messages], snapshot)

        return should_compact

    def _loop_compact(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        self._emit_hook(HookRecord("on_compact", metadata={"messages": len(messages)}))
        if self.compactor is None:
            return messages
        dict_messages = [to_dict(m) for m in messages]
        self.compactor(dict_messages)
        # compactor 通过 dict 操作，返回值由 harness 层管理。
        return messages

    def _loop_prepare_next_turn(
        self, snapshot: TurnSnapshot
    ) -> Callable[[], AgentLoopTurnUpdate | None]:
        def prepare_next_turn() -> AgentLoopTurnUpdate | None:
            self._progress_steps_without_update += 1
            if self._progress_steps_without_update >= 5:
                self._progress_steps_without_update = 0
                self.steer(
                    UserMessage(
                        content="<reminder>You have gone several turns without updating task progress. "
                        "Use update_task or save_task_progress to record progress before continuing.</reminder>"
                    )
                )

            if self._current_mode == "plan":
                self._plan_enter_step += 1
                if self._plan_enter_step >= self._max_plan_turns:
                    self._plan_enter_step = 0
                    self._current_mode = "act"
                    self._agent.update_tools(
                        self._tools_for_mode(snapshot.registry, "act")
                    )
                    self.steer(
                        SystemMessage(
                            content=(
                                "<plan-timeout>\n"
                                "Plan Mode timed out after reaching the maximum number "
                                "of investigation turns. Returning to Act Mode.\n"
                                "</plan-timeout>"
                            )
                        )
                    )
            return None

        return prepare_next_turn

    def _loop_is_tool_productive(
        self, mode: ExecutionMode, snapshot: TurnSnapshot
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
                snapshot.tool_map,
            )

        return is_tool_productive

    def _loop_before_tool(
        self, mode: ExecutionMode, snapshot: TurnSnapshot
    ) -> Callable[[BeforeToolCallContext, Any], BeforeToolCallResult | None]:
        def before_tool(
            ctx: BeforeToolCallContext, _signal: Any
        ) -> BeforeToolCallResult | None:
            tool_call = ctx.tool_call
            args = ctx.args
            action_input = stringify_tool_input(args)

            if self._plan_pending_confirmation and self._plan_confirmation_required(
                ctx
            ):
                self._plan_pending_confirmation = False
                return BeforeToolCallResult(
                    block=True,
                    reason=(
                        f"tool {tool_call.name} requires plan confirmation. "
                        "Present the plan to the user for approval before executing write tools."
                    ),
                )

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
                approval = self._request_tool_approval(tool_call.name, args, snapshot)
                if approval is not None:
                    return approval

            self._emit_hook(
                HookRecord("pre_tool", tool=tool_call.name, input=action_input)
            )
            return None

        return before_tool

    def _request_tool_approval(
        self, tool_name: str, args: dict[str, Any], snapshot: TurnSnapshot
    ) -> BeforeToolCallResult | None:
        if (
            snapshot.approval_callback is None
            or snapshot.tool_map.get(tool_name) is None
        ):
            return BeforeToolCallResult(
                block=True,
                reason=f"tool requires approval: {tool_name}",
            )
        hitl = snapshot.approval_callback(snapshot.tool_map[tool_name], args)
        if hitl.decision == "deny":
            return BeforeToolCallResult(
                block=True,
                reason=f"tool {tool_name} denied by user",
            )
        return None

    PROGRESS_TOOL_NAMES = frozenset(
        {
            "save_task_progress",
            "resume_task_progress",
            "update_task",
            "create_task",
        }
    )

    def _loop_after_tool(
        self, snapshot: TurnSnapshot
    ) -> Callable[[AfterToolCallContext, Any], AfterToolCallResult | None]:
        def after_tool(
            ctx: AfterToolCallContext, _signal: Any
        ) -> AfterToolCallResult | None:
            if ctx.tool_call.name in self.PROGRESS_TOOL_NAMES:
                self._progress_steps_without_update = 0
            action_input = stringify_tool_input(ctx.args)
            result_content_text = _tool_result_text(ctx)

            self._emit_tool_hook(ctx, action_input, result_content_text)
            self._emit_audit_record(ctx, action_input, result_content_text, snapshot)
            return None

        return after_tool

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
        snapshot: TurnSnapshot,
    ) -> None:
        if self.audit_logger is None:
            return
        tool_call = ctx.tool_call
        self.audit_logger(
            AuditRecord(
                session_id=self.session_id,
                tool=tool_call.name,
                static_risk=snapshot.tool_map.get(
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

    def _should_compact(
        self, messages: list[dict[str, Any]], snapshot: TurnSnapshot
    ) -> bool:
        if self.compactor is None:
            return False
        if self.manual_compact_requested and self.manual_compact_requested():
            return True
        return (
            snapshot.config.compact_threshold > 0
            and len(messages) > snapshot.config.compact_threshold
        ) or (
            snapshot.config.compact_token_threshold > 0
            and estimate_message_tokens(messages)
            > snapshot.config.compact_token_threshold
        )

    def _turn_context_messages(
        self,
        question: str,
        mode: ExecutionMode,
        snapshot: TurnSnapshot,
    ) -> list[AgentMessage]:
        self._current_mode = mode
        typed: list[AgentMessage] = []
        notice = mode_notice(mode)
        if snapshot.runtime_context_provider is not None:
            parts = snapshot.runtime_context_provider(question)
            if notice:
                parts.append(notice)
            if parts:
                typed.append(SystemMessage(content="\n\n".join(p for p in parts if p)))
        elif notice:
            typed.append(SystemMessage(content=notice))

        return typed

    def _save_turn_to_history(self, messages: list[AgentMessage]) -> None:
        """保存本轮对话到历史。

        包括：用户消息、助手消息、工具调用和工具结果。
        """
        self._session.append(messages)

    def _turn_snapshot(self) -> TurnSnapshot:
        """冻结当前 turn 依赖的配置和工具引用。"""
        registry = tuple(self.registry)
        return TurnSnapshot(
            config=deepcopy(self.config),
            registry=registry,
            tool_map={tool.name: tool for tool in registry},
            approval_callback=self.approval_callback,
            permission_policy=self.permission_policy,
            provider=self.provider,
            runtime_context_provider=self.runtime_context_provider,
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


def _messages_from_run_state(run_state: RunState) -> list[AgentMessage]:
    """从可序列化运行状态恢复模型可见消息。"""
    messages: list[AgentMessage] = []
    for item in run_state.messages:
        message = _message_from_dict(item)
        if message is not None:
            messages.append(message)
    return messages


def _message_from_dict(item: dict[str, Any]) -> AgentMessage | None:
    role = str(item.get("role", ""))
    if role == "system":
        return SystemMessage(content=str(item.get("content", "")))
    if role == "user":
        return UserMessage(content=str(item.get("content", "")))
    if role == "assistant":
        return _assistant_from_dict(item)
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=str(item.get("tool_call_id", "")),
            content=str(item.get("content", "")),
        )
    return None


def _assistant_from_dict(item: dict[str, Any]) -> AssistantMessage:
    content: list[ContentBlock] = []
    text = item.get("content")
    if isinstance(text, str) and text:
        content.append(TextContent(text=text))
    tool_calls = item.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            parsed = _tool_call_from_dict(tool_call)
            if parsed is not None:
                content.append(parsed)
    return AssistantMessage(content=content)


def _tool_call_from_dict(item: object) -> ToolCallContent | None:
    if not isinstance(item, dict):
        return None
    function = item.get("function", {})
    if not isinstance(function, dict):
        return None
    name = str(function.get("name", "")).strip()
    tool_call_id = str(item.get("id", "")).strip()
    if not name or not tool_call_id:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = {}
        arguments = decoded
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCallContent(id=tool_call_id, name=name, arguments=arguments)
