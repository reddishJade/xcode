"""AgentHarness — 通用 agent 运行时，领域无关。

与 Agent 核心循环的关系：负责 agent 生命周期管理（创建、配置、运行、
清理），但本身不包含任何编码领域概念。编码特定行为通过 protected
方法覆盖由子类 CodingAgentHarness 提供。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from copy import deepcopy

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
    TurnSnapshot,
    build_turn_snapshot,
    build_loop_config,
    record_last_prompt_tokens,
    resolve_permission_policy,
    GateConfig,
)
from .events import (
    _StreamTranslationState,
    _translate_event,
    CodingAgentHarnessEvent,
)
from .fallback import _FallbackWithRetryPrimary
from .message_codec import messages_from_run_state
from .result import (
    _build_structured_result,
    _final_event,
    RunState,
    CodingAgentHarnessResult,
)
from .run_control import (
    ActiveRunHandle,
    BusyMessageMode,
    SessionRunController,
    SubmitOutcome,
)
from .tool_gate import ToolGate
from ._mode_protocol import ToolGateMode
from ..config import AgentConfig, RequestHygieneConfig
from xcode.ai.events import ToolCall
from ..observability.permissions import PermissionDecision
from ..observability import HookRecord, RuntimeCorrelation
from ..observability.permission_model import GrantStore
from ...agent.types import ApprovalCallback

_PROMPT_VERSION_CACHE: str | None = None


def _get_prompt_version() -> str:
    global _PROMPT_VERSION_CACHE
    if _PROMPT_VERSION_CACHE is None:
        from .prompting.identity import PROMPT_VERSION as _v

        _PROMPT_VERSION_CACHE = _v
    return _PROMPT_VERSION_CACHE or "unknown"


__all__ = ["AgentHarness"]


class _DefaultToolGateMode:
    """AgentHarness 默认 mode：始终为 "act"，check_call 返回 "ask"。"""

    @property
    def current_mode(self) -> str:
        return "act"

    def check_call(self, call: ToolCall) -> PermissionDecision:
        return "ask"


class AgentHarness:
    """通用 agent 运行时包装。

    不感知执行模式（plan/build/act）、技能系统、记忆反馈等编码领域概念。
    这些由子类 CodingAgentHarness 提供。
    """

    def __init__(
        self,
        provider: ModelProvider,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig | None = None,
        gate: GateConfig | None = None,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        gate = gate or GateConfig()
        runtime = runtime or AgentRuntimeConfig()
        config = config or runtime.config

        self.provider: ModelProvider = provider
        if runtime.fallback_provider is not None:
            self.provider = _FallbackWithRetryPrimary(
                provider, runtime.fallback_provider
            )
        self.project_root = runtime.project_root
        self._runtime = runtime
        self._registry = registry
        self.config = config
        self.compactor = runtime.compactor
        self._compact_controller = runtime.compact_controller
        self.runtime_context_provider = runtime.runtime_context_provider
        self.cancellation_token = runtime.cancellation_token or CancellationToken()
        self.request_hygiene = runtime.request_hygiene or RequestHygieneConfig()
        self._correlation = gate.correlation or RuntimeCorrelation(gate.session_id)
        self._last_prompt_tokens: int | None = None

        self._hook_manager = gate.hook_manager
        resolved_permission_policy = resolve_permission_policy(
            runtime.project_root, gate.permission_policy
        )
        self.permission_policy = resolved_permission_policy
        self.restricted_dirs = gate.restricted_dirs
        self.hook_constraint_providers = gate.hook_constraint_providers
        self._gate = ToolGate(
            mode_state=self._build_gate_mode(),
            approval_callback=gate.approval_callback,
            permission_policy=resolved_permission_policy,
            hook_manager=gate.hook_manager,
            external_hook_runner=gate.external_hook_runner,
            external_hooks_subagent=gate.external_hooks_subagent,
            external_hooks_cwd=gate.external_hooks_cwd,
            correlation=self._correlation,
            audit_logger=gate.audit_logger,
            session_id=gate.session_id,
            restricted_dirs=gate.restricted_dirs,
            hook_constraint_providers=gate.hook_constraint_providers,
            project_root=runtime.project_root,
            session_grant_store=gate.session_grant_store,
            session_grant_store_provider=gate.session_grant_store_provider,
            permanent_grant_store=gate.permanent_grant_store,
            user_rulesets=gate.user_rulesets,
        )
        self.audit_logger = gate.audit_logger
        self._history: list[AgentMessage] = []
        self._resumed_notice: str | None = None
        self._run_controller = SessionRunController(gate.session_id)

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
        snapshot: TurnSnapshot,
    ) -> list[AgentMessage]:
        """构建 turn 上下文消息。子类可添加模式通知、记忆概览等。"""
        parts: list[str] = []
        if snapshot.runtime_context_provider is not None:
            parts = list(snapshot.runtime_context_provider(question))
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

    def _build_result(
        self, visible_result: object, max_steps: int
    ) -> CodingAgentHarnessResult:
        """构建 turn 结果。子类可覆盖以注入 current_mode 等。"""
        return _build_structured_result(visible_result, max_steps)  # type: ignore[arg-type]

    def _post_run(self, final: CodingAgentHarnessResult) -> None:
        """turn 完成后的子类钩子。例如记忆反馈。"""
        pass

    # ── 公共 API ──

    @property
    def registry(self) -> tuple[ToolSpec, ...]:
        """返回当前工具注册表快照。"""
        return self._registry

    @property
    def tool_map(self) -> dict[str, ToolSpec]:
        """按名称返回当前工具映射。"""
        return {tool.name: tool for tool in self.registry}

    def steer(self, msg: AgentMessage) -> None:
        self._try_runtime_steer(msg)

    def _try_runtime_steer(self, msg: AgentMessage) -> bool:
        """注入运行时内部消息，包括 reminder 和 SystemMessage。"""
        handle = self.active_run()
        return bool(handle and handle.steer(msg).accepted)

    def try_steer(self, msg: AgentMessage) -> bool:
        """尝试向当前活跃 run 注入外部用户消息。"""
        if not isinstance(msg, UserMessage):
            return False
        return self._try_runtime_steer(msg)

    def follow_up(self, msg: UserMessage) -> bool:
        """把用户消息排入当前 run 结束后的新 run。"""
        return self._run_controller.submit(msg, BusyMessageMode.FOLLOW_UP).accepted

    def submit_busy_message(
        self,
        msg: UserMessage,
        mode: BusyMessageMode = BusyMessageMode.STEER,
    ) -> SubmitOutcome:
        """按照指定 busy policy 提交运行时用户消息。"""
        return self._run_controller.submit(msg, mode)

    def active_run(self) -> ActiveRunHandle | None:
        """返回当前 session 的 active run handle。"""
        return self._run_controller.active_run()

    def interrupt(self, reason: str = "interrupted by user") -> bool:
        """请求取消当前 run，但在其完整退出前保留 active identity。"""
        handle = self.active_run()
        return bool(handle and handle.interrupt(reason).accepted)

    def take_follow_up(self) -> UserMessage | None:
        """当前 run 完成后取出下一条 session-level follow-up。"""
        return self._run_controller.take_follow_up()

    def request_compaction(self) -> None:
        if self._compact_controller is not None:
            self._compact_controller.request()

    def clear_history(self) -> None:
        self._history = []
        self._gate.clear_session_grants()
        self._reset_provider_conversation_state()

    @property
    def approval_callback(self) -> ApprovalCallback | None:
        """返回当前 HITL 审批回调。"""
        return self._gate.approval_callback

    @approval_callback.setter
    def approval_callback(self, value: ApprovalCallback | None) -> None:
        """更新后续工具执行使用的 HITL 审批回调。"""
        self._gate.set_approval_callback(value)

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
        self._run_controller.session_id = value

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

    def run(self, question: str) -> CodingAgentHarnessResult:
        return run_coro_sync(self.arun(question))

    async def run_async(self, question: str) -> CodingAgentHarnessResult:
        return await self.arun(question)

    async def arun(self, question: str) -> CodingAgentHarnessResult:
        result: CodingAgentHarnessResult | None = None
        async for event in self.arun_stream(question):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(self, question: str) -> Iterator[CodingAgentHarnessEvent]:
        yield from aiter_to_sync_iter(
            self.arun_stream(question), self.cancellation_token
        )

    async def arun_stream(
        self, question: str
    ) -> AsyncIterator[CodingAgentHarnessEvent]:
        from .tool_hooks import emit_hook as _emit_hook

        snapshot = build_turn_snapshot(
            self.config,
            tuple(self.registry),
            self.provider,
            self.runtime_context_provider,
        )
        registry_snapshot = snapshot.registry
        active_registry = self._build_active_registry(registry_snapshot)
        context_messages = self._build_context_messages(question, snapshot)
        self._resumed_notice = None
        history_messages = context_messages + self.history_messages()
        turn_messages: list[AgentMessage] = [UserMessage(content=question)]

        turn_agent = Agent(self._gate.adapt_tools(active_registry))
        loop_config = build_loop_config(
            snapshot=snapshot,
            gate=self._gate,
            registry=active_registry,
            compactor=self.compactor,
            manual_compact_requested=(
                self._compact_controller.consume if self._compact_controller else None
            ),
            request_hygiene=self.request_hygiene,
            compact_controller=self._compact_controller,
            last_prompt_tokens=self._last_prompt_tokens,
            steer=self.steer,
            emit_hook=lambda rec: _emit_hook(self._hook_manager, rec),
            get_prompt_version=_get_prompt_version,
            project_root=self.project_root,
            prompt_instructions=self._runtime.prompt_instructions,
            correlation=self._correlation,
            **self._build_loop_config_extras(),
        )

        run_handle = self._run_controller.begin_run(turn_agent, self.cancellation_token)
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
            ):
                translated = _translate_event(event, translation_state)
                if translated is not None:
                    for te in (
                        translated if isinstance(translated, list) else [translated]
                    ):
                        yield te

            result = turn_agent.last_result
            assert result is not None

            self._history.extend(result.messages)
            self._last_prompt_tokens = record_last_prompt_tokens(result.messages)

            visible_result = (
                result.model_copy(
                    update={"messages": context_messages + result.messages}
                )
                if context_messages
                else result
            )
            final = self._build_result(visible_result, snapshot.config.max_steps)
            self._post_run(final)
            yield _final_event(
                result.steps,
                final,
                self._correlation.snapshot(),
            )
        finally:
            run_handle.begin_finishing()
            unconsumed_steers = turn_agent.close_steering()
            self._run_controller.complete_run(run_handle, unconsumed_steers)

    # ── 内部 ──

    def _reset_provider_conversation_state(self) -> None:
        reset = getattr(self.provider, "reset_conversation_state", None)
        if callable(reset):
            reset()
