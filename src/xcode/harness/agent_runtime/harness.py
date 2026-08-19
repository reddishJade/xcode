"""AgentHarness — 通用 agent 运行时，领域无关。

与 Agent 核心循环的关系：负责 agent 生命周期管理（创建、配置、运行、
清理），但本身不包含任何编码领域概念。编码特定行为通过 protected
方法覆盖由子类 CodingAgentHarness 提供。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from copy import deepcopy
from threading import Lock

from xcode.ai.providers.base import ModelProvider

from ...agent.agent import Agent
from ...agent.messages import (
    AgentMessage,
    SystemMessage,
    UserMessage,
)
from ...agent.types import ToolSpec
from .agent_helpers import aiter_to_sync_iter, run_coro_sync
from .cancellation import CancellationToken
from .config import (
    AgentRuntimeConfig,
    build_loop_config,
    record_last_prompt_tokens,
)
from .composition import AgentComposition
from .events import (
    _StreamTranslationState,
    _translate_event,
    AgentHarnessEvent,
)
from .fallback import _FallbackWithRetryPrimary
from .message_codec import messages_from_run_state
from .result import (
    _build_structured_result,
    _final_event,
    RunState,
    AgentHarnessResult,
)
from .run_control import (
    ActiveRunHandle,
    BusyMessageMode,
    SessionRunController,
    SubmitOutcome,
)
from .tool_gate import ToolGate
from ._mode_protocol import ToolGateMode
from xcode.ai.events import ToolCall
from ..security.permissions import PermissionDecision, PermissionPolicy
from ..security.approval import ApprovalPolicy, ApprovalsReviewer
from ..observability import HookRecord, RuntimeCorrelation
from ..security.permission_model import GrantStore
from ...agent.types import ApprovalCallback

_PROMPT_VERSION_CACHE: str | None = None


def _get_prompt_version() -> str:
    global _PROMPT_VERSION_CACHE
    if _PROMPT_VERSION_CACHE is None:
        from .prompting.identity import prompt_version

        _PROMPT_VERSION_CACHE = prompt_version()
    return _PROMPT_VERSION_CACHE or "runtime"


__all__ = ["AgentHarness"]


class _DefaultToolGateMode:
    """AgentHarness 默认 mode：始终为 "act"，check_call 返回 "ask"。"""

    @property
    def current_mode(self) -> str:
        return "act"

    @property
    def approvals_reviewer(self) -> ApprovalsReviewer:
        return "user"

    def check_call(self, call: ToolCall) -> PermissionDecision:
        return "ask"


class AgentHarness:
    """通用 agent 运行时包装。

    不感知执行模式（plan/build/act）、技能系统、记忆反馈等编码领域概念。
    这些由子类 CodingAgentHarness 提供。
    """

    def __init__(
        self,
        composition: AgentComposition,
        runtime: AgentRuntimeConfig,
    ) -> None:
        self._composition_lock = Lock()
        self._composition = composition
        self._provider = _provider_for(composition)
        gate = composition.gate
        gate_runtime = runtime.gate
        self.project_root = runtime.project_root
        self._runtime = runtime
        self.compactor = runtime.compactor
        self._compact_controller = runtime.compact_controller
        self.cancellation_token = runtime.cancellation_token or CancellationToken()
        supplied_gate = runtime.gate_instance
        self._correlation = (
            supplied_gate.correlation
            if supplied_gate is not None
            else gate_runtime.correlation or RuntimeCorrelation(gate_runtime.session_id)
        )
        self._last_prompt_tokens: int | None = None

        self._hook_manager = gate_runtime.hook_manager
        self.external_directories = gate.external_directories
        self.sensitive_path_overrides = gate.sensitive_path_overrides
        self.hook_constraint_providers = gate.hook_constraint_providers
        self._gate = supplied_gate or ToolGate(
            mode_state=self._build_gate_mode(),
            user_approval_callback=gate_runtime.user_approval_callback,
            auto_approval_callback=gate_runtime.auto_approval_callback,
            permission_policy=gate.permission_policy,
            approval_policy=gate.approval_policy,
            hook_manager=gate_runtime.hook_manager,
            external_hook_runner=gate_runtime.external_hook_runner,
            external_hooks_subagent=gate_runtime.external_hooks_subagent,
            external_hooks_cwd=gate_runtime.external_hooks_cwd,
            correlation=self._correlation,
            audit_logger=gate_runtime.audit_logger,
            session_id=gate_runtime.session_id,
            restricted_dirs=gate.restricted_dirs,
            hook_constraint_providers=gate.hook_constraint_providers,
            project_root=runtime.project_root,
            external_directories=gate.external_directories,
            sensitive_path_overrides=gate.sensitive_path_overrides,
            session_grant_store=gate_runtime.session_grant_store,
            session_grant_store_provider=gate_runtime.session_grant_store_provider,
            permanent_grant_store=gate_runtime.permanent_grant_store,
            user_rulesets=gate.user_rulesets,
            default_mode_rulesets=gate.default_mode_rulesets,
            mode_fallbacks=gate.mode_fallbacks,
            shell_unresolved_policies=gate.shell_unresolved_policies,
            tool_path_extractors=gate.tool_path_extractors,
        )
        self.audit_logger = gate_runtime.audit_logger
        self._history: list[AgentMessage] = []
        self._resumed_notice: str | None = None
        self._run_controller = SessionRunController(runtime.session_inbox)

    # ── 可被子类覆盖的扩展点 ──

    def _build_gate_mode(self) -> ToolGateMode:
        """返回实现 ToolGateMode 协议的 mode 对象。

        子类可返回一个带有 current_mode 和 check_call() 的 mode state。
        基类返回一个提供 mode 名称为 "act" 的简单对象。
        """
        return _DefaultToolGateMode()

    def _build_active_registry(
        self, registry: tuple[ToolSpec, ...]
    ) -> tuple[ToolSpec, ...]:
        """返回当前 turn 使用的工具集。子类可实现模式过滤。"""
        return registry

    def _build_context_messages(
        self,
        question: str,
        composition: AgentComposition,
    ) -> list[AgentMessage]:
        """构建 turn 上下文消息。子类可添加模式通知、记忆概览等。"""
        parts: list[str] = []
        if composition.runtime_context_provider is not None:
            parts = list(composition.runtime_context_provider(question))
        if self._resumed_notice is not None:
            parts.append(
                f"<session-notices>\n{self._resumed_notice}\n</session-notices>"
            )
        if parts:
            return [SystemMessage(content="\n\n".join(p for p in parts if p))]
        return []

    def _build_loop_config_extras(self) -> dict:
        """子类可返回额外的 build_loop_config 参数。"""
        return {}

    def _build_result(self, visible_result: object) -> AgentHarnessResult:
        """构建 turn 结果。子类可覆盖以注入 current_mode 等。"""
        return _build_structured_result(visible_result)  # type: ignore[arg-type]

    def _post_run(self, final: AgentHarnessResult) -> None:
        """turn 完成后的子类钩子。例如记忆反馈。"""
        pass

    # ── 公共 API ──

    @property
    def composition(self) -> AgentComposition:
        """返回当前已发布的不可变 composition generation。"""
        with self._composition_lock:
            return self._composition

    @property
    def provider(self) -> ModelProvider:
        """返回当前 composition 对应的有效 provider。"""
        with self._composition_lock:
            return self._provider

    @property
    def registry(self) -> tuple[ToolSpec, ...]:
        """返回当前工具注册表快照。"""
        return self.composition.registry

    @property
    def permission_policy(self) -> PermissionPolicy | None:
        return self.composition.gate.permission_policy

    @property
    def approval_policy(self) -> ApprovalPolicy:
        """返回 ask 是否可提交审批。"""
        return self._gate.approval_policy

    @property
    def approvals_reviewer(self) -> ApprovalsReviewer:
        """返回 ask 当前交给人还是独立 reviewer。"""
        return self._gate.approvals_reviewer

    @property
    def restricted_dirs(self) -> tuple[str, ...]:
        return self.composition.gate.restricted_dirs

    @property
    def tool_map(self) -> dict[str, ToolSpec]:
        """按名称返回当前工具映射。"""
        return {tool.name: tool for tool in self.registry}

    def replace_primary_provider(self, provider: ModelProvider) -> str:
        """原子发布使用新主 provider 的 composition generation。"""
        with self._composition_lock:
            if self.active_run() is not None:
                raise RuntimeError("cannot replace composition during an active run")
            composition = self._composition.with_primary_provider(provider)
            self._composition = composition
            self._provider = _provider_for(composition)
        return composition.generation_id

    def replace_permission_policy(self, policy: PermissionPolicy | None) -> str:
        """原子发布使用新静态权限策略的 composition generation。"""
        with self._composition_lock:
            if self.active_run() is not None:
                raise RuntimeError("cannot replace composition during an active run")
            composition = self._composition.with_permission_policy(policy)
            self._composition = composition
            self._gate.set_permission_policy(composition.gate.permission_policy)
        return composition.generation_id

    def inject(self, msg: AgentMessage) -> SubmitOutcome:
        """注入运行时内部消息，包括 reminder 和 SystemMessage。"""
        return self._run_controller.inject_runtime(msg)

    def steer(
        self,
        msg: UserMessage,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome:
        """把用户输入调度到当前 run 的下一模型边界。"""
        return self._run_controller.submit(
            msg,
            BusyMessageMode.STEER,
            display_text=display_text,
        )

    def followup(
        self,
        msg: UserMessage,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome:
        """把用户输入排入独立的后续 turn。"""
        return self._run_controller.submit(
            msg,
            BusyMessageMode.FOLLOW_UP,
            display_text=display_text,
        )

    def submit_busy_message(
        self,
        msg: UserMessage,
        mode: BusyMessageMode = BusyMessageMode.STEER,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome:
        """按照指定 busy policy 提交运行时用户消息。"""
        return self._run_controller.submit(msg, mode, display_text=display_text)

    def active_run(self) -> ActiveRunHandle | None:
        """返回当前 session 的 active run handle。"""
        return self._run_controller.active_run()

    def interrupt(self, reason: str = "interrupted by user") -> bool:
        """请求取消当前 run，但在其完整退出前保留 active identity。"""
        handle = self.active_run()
        if handle is None:
            return False
        outcome = handle.interrupt(reason)
        return outcome is not None

    def has_pending_input(self) -> bool:
        """返回 durable inbox 是否存在需要启动的新输入。"""
        return self._run_controller.has_waking_input()

    def request_compaction(self) -> None:
        if self._compact_controller is not None:
            self._compact_controller.request()

    def clear_history(self) -> None:
        self._history = []
        self._gate.clear_session_grants()
        self._reset_provider_conversation_state()

    @property
    def current_approval_callback(self) -> ApprovalCallback | None:
        """返回当前 execution mode 实际使用的审批回调。"""
        return self._gate.current_approval_callback

    @property
    def user_approval_callback(self) -> ApprovalCallback | None:
        return self._gate.user_approval_callback

    @user_approval_callback.setter
    def user_approval_callback(self, value: ApprovalCallback | None) -> None:
        """更新 Act 等人工审批模式使用的回调。"""
        self._gate.set_user_approval_callback(value)

    @property
    def auto_approval_callback(self) -> ApprovalCallback | None:
        return self._gate.auto_approval_callback

    def set_session_grant_store_provider(
        self,
        provider: Callable[[], GrantStore | None] | None,
    ) -> None:
        """设置当前会话的 session grant store provider。"""
        self._gate.set_session_grant_store_provider(provider)

    def set_permanent_grant_store(self, store: GrantStore | None) -> None:
        """设置 permanent grant store。"""
        self._gate.set_permanent_grant_store(store)

    @property
    def session_id(self) -> str:
        return self._gate.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._gate.session_id = value
        self._correlation.session_id = value
        self._run_controller.reload()

    def load_history(self, messages: list[AgentMessage]) -> None:
        self._history = deepcopy(messages)
        self._post_load_history(messages)
        if not messages:
            self._gate.clear_session_grants()
        self._reset_provider_conversation_state()

    def _post_load_history(self, messages: list[AgentMessage]) -> None:
        """load_history 的子类钩子，例如恢复技能激活。"""
        pass

    def set_resumed_notice(self, notice: str) -> None:
        self._resumed_notice = notice

    def load_run_state(self, run_state: RunState) -> None:
        self._history = messages_from_run_state(run_state)
        self._reset_provider_conversation_state()

    def history_messages(self) -> list[AgentMessage]:
        return list(self._history)

    def run(self, question: str) -> AgentHarnessResult:
        return run_coro_sync(self.arun(question))

    async def run_async(self, question: str) -> AgentHarnessResult:
        return await self.arun(question)

    async def arun(self, question: str) -> AgentHarnessResult:
        result: AgentHarnessResult | None = None
        async for event in self.arun_stream(question):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(self, question: str) -> Iterator[AgentHarnessEvent]:
        yield from aiter_to_sync_iter(
            self.arun_stream(question), self.cancellation_token
        )

    async def arun_stream(
        self,
        question: str | None,
        *,
        display_question: str | None = None,
    ) -> AsyncIterator[AgentHarnessEvent]:
        if question is not None:
            self.followup(
                UserMessage(content=question),
                display_text=display_question,
            )

        if not self._run_controller.has_waking_input():
            return
        with self._composition_lock:
            run_handle = self._run_controller.begin_run(self.cancellation_token)
        try:
            turn_messages = self._run_controller.claim_initial(run_handle)
            if not turn_messages:
                return
            question_text = _turn_text(turn_messages)
            async for event in self._arun_claimed_turn(
                question_text,
                turn_messages,
                run_handle,
            ):
                yield event
        finally:
            if self._run_controller.active_run() is run_handle:
                run_handle.begin_finishing()
                self._run_controller.complete_run(run_handle)

    async def _arun_claimed_turn(
        self,
        question: str,
        turn_messages: list[AgentMessage],
        run_handle: ActiveRunHandle,
    ) -> AsyncIterator[AgentHarnessEvent]:
        from .tool_hooks import emit_hook as _emit_hook

        with self._composition_lock:
            composition = self._composition
            provider = self._provider
        registry_snapshot = composition.registry
        active_registry = self._build_active_registry(registry_snapshot)
        context_messages = self._build_context_messages(question, composition)
        self._resumed_notice = None
        history_messages = self.history_messages()
        turn_agent = Agent(self._gate.adapt_tools(active_registry))

        def inject_runtime(message: AgentMessage) -> None:
            self.inject(message)

        loop_config = build_loop_config(
            composition=composition,
            provider=provider,
            gate=self._gate,
            registry=active_registry,
            compactor=self.compactor,
            manual_compact_requested=(
                self._compact_controller.consume if self._compact_controller else None
            ),
            compact_controller=self._compact_controller,
            last_prompt_tokens=self._last_prompt_tokens,
            steer=inject_runtime,
            emit_hook=lambda rec: _emit_hook(self._hook_manager, rec),
            get_prompt_version=_get_prompt_version,
            correlation=self._correlation,
            **self._build_loop_config_extras(),
        )

        try:
            self.cancellation_token.reset()
            self._correlation.reset(self.session_id)

            current = self._correlation.snapshot()
            _emit_hook(
                self._hook_manager,
                HookRecord(
                    "before_agent_start",
                    metadata={"question": question},
                    timestamp=current.timestamp,
                    session_id=current.session_id,
                    turn_id=current.turn_id,
                    request_id=current.request_id,
                ),
            )

            translation_state = _StreamTranslationState(correlation=self._correlation)
            async for event in turn_agent.run_stream(
                turn_messages,
                loop_config,
                signal=self.cancellation_token,
                history=history_messages,
                request_prefix=context_messages,
                step_input=run_handle.claim_step_input,
                finish_step_input=run_handle.finish_step_input,
                reopen_step_input=run_handle.reopen_step_input,
            ):
                translated = _translate_event(event, translation_state)
                if translated is not None:
                    for te in (
                        translated if isinstance(translated, list) else [translated]
                    ):
                        yield te

            result = turn_agent.last_result
            assert result is not None

            self._history = list(result.surface)
            self._last_prompt_tokens = record_last_prompt_tokens(result.surface)

            visible_result = (
                result.model_copy(
                    update={"messages": context_messages + result.messages}
                )
                if context_messages
                else result
            )
            final = self._build_result(visible_result)
            self._post_run(final)
            yield _final_event(
                result.steps,
                final,
                self._correlation.snapshot(),
            )
        finally:
            run_handle.begin_finishing()
            self._run_controller.complete_run(run_handle)

    # ── 内部 ──

    def _reset_provider_conversation_state(self) -> None:
        reset = getattr(self.provider, "reset_conversation_state", None)
        if callable(reset):
            reset()


def _turn_text(messages: list[AgentMessage]) -> str:
    """为上下文收集器提取本次 claim 的文本查询。"""
    from ...agent.types import TextContent

    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
            continue
        parts.extend(
            block.text
            for block in content or []
            if isinstance(block, TextContent) and block.text
        )
    return "\n\n".join(parts)


def _provider_for(composition: AgentComposition) -> ModelProvider:
    fallback = composition.fallback_provider
    if fallback is None:
        return composition.primary_provider
    return _FallbackWithRetryPrimary(composition.primary_provider, fallback)
