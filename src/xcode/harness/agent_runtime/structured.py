"""StructuredAgent — harness 层对 agent/Agent 的适配。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import replace

from xcode.ai.providers.protocol import StreamProvider

from ...agent.agent import Agent
from ...agent.messages import AgentMessage, UserMessage
from ...agent.protocols import AgentTool
from .agent_helpers import aiter_to_sync_iter, run_coro_sync
from .cancellation import CancellationToken
from .config import (
    AgentRuntimeConfig,
    build_loop_config,
    build_turn_context_messages,
    build_turn_snapshot,
    GateConfig,
    record_last_prompt_tokens,
    resolve_permission_policy,
)
from .events import (
    _StreamTranslationState,
    _translate_event,
    StructuredAgentEvent,
)
from .execution_modes import ExecutionModeState, policy_for_mode
from .fallback import _FallbackWithRetryPrimary
from .history_manager import HistoryManager
from .result import (
    _build_structured_result,
    _final_event,
    RunState,
    StructuredAgentResult,
)
from .tool_gate import ToolGate
from ..config import AgentConfig, ExecutionMode, RequestHygieneConfig
from ..observability import HookRecord
from ..skills import ApprovalCallback, ToolSpec

_PROMPT_VERSION_CACHE: str | None = None


def _get_prompt_version() -> str:
    global _PROMPT_VERSION_CACHE
    if _PROMPT_VERSION_CACHE is None:
        from .prompting.identity import PROMPT_VERSION as _v  # pyright: ignore

        _PROMPT_VERSION_CACHE = _v
    return _PROMPT_VERSION_CACHE or "unknown"


__all__ = ["StructuredAgent"]


class StructuredAgent:
    def __init__(
        self,
        provider: StreamProvider,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig | None = None,
        gate: GateConfig | None = None,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        gate = gate or GateConfig()
        runtime = runtime or AgentRuntimeConfig()
        config = config or runtime.config

        self.provider: StreamProvider = provider
        if runtime.fallback_provider is not None:
            self.provider = _FallbackWithRetryPrimary(
                provider, runtime.fallback_provider
            )
        self.project_root = runtime.project_root
        self.registry = registry
        self.tool_map = {t.name: t for t in registry}
        self.config = config
        self.compactor = runtime.compactor
        self._compact_controller = runtime.compact_controller
        self.runtime_context_provider = runtime.runtime_context_provider
        self.cancellation_token = runtime.cancellation_token or CancellationToken()
        self.request_hygiene = runtime.request_hygiene or RequestHygieneConfig()
        self._last_prompt_tokens: int | None = None

        self._hook_manager = gate.hook_manager
        self._mode = ExecutionModeState()
        resolved_permission_policy = resolve_permission_policy(
            runtime.project_root, gate.permission_policy
        )
        self.permission_policy = resolved_permission_policy
        self.restricted_dirs = gate.restricted_dirs
        self.allowlist_mode = gate.allowlist_mode
        self._gate = ToolGate(
            mode_state=self._mode,
            approval_callback=gate.approval_callback,
            permission_policy=resolved_permission_policy,
            high_risk_requires_approval=gate.high_risk_requires_approval,
            hook_manager=gate.hook_manager,
            audit_logger=gate.audit_logger,
            session_id=gate.session_id,
            restricted_dirs=gate.restricted_dirs,
            allowlist_mode=gate.allowlist_mode,
        )
        self.audit_logger = gate.audit_logger
        self._history = HistoryManager()
        self._resumed_notice: str | None = None

    # ── 公共 API ──

    def steer(self, msg: AgentMessage) -> None:
        self._agent.steer(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        self._agent.follow_up(msg)

    def request_compaction(self) -> None:
        if self._compact_controller is not None:
            self._compact_controller.request()

    def clear_history(self) -> None:
        self._history.clear()
        self._reset_provider_conversation_state()

    @property
    def approval_callback(self) -> ApprovalCallback | None:
        """返回当前 HITL 审批回调。"""
        return self._gate.approval_callback

    @approval_callback.setter
    def approval_callback(self, value: ApprovalCallback | None) -> None:
        """更新后续工具执行使用的 HITL 审批回调。"""
        self._gate.set_approval_callback(value)

    def load_history(self, messages: list[AgentMessage]) -> None:
        self._history.load(messages)
        self._reset_provider_conversation_state()

    def set_resumed_notice(self, notice: str) -> None:
        self._resumed_notice = notice

    def load_run_state(self, run_state: RunState) -> None:
        self._history.load_run_state(run_state)
        self._reset_provider_conversation_state()
        restored = self._history.restore_mode(run_state)
        if restored is not None:
            self._mode.set_mode(restored)

    def history_messages(self) -> list[AgentMessage]:
        return self._history.messages()

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
        from .tool_hooks import emit_hook as _emit_hook

        snapshot = build_turn_snapshot(
            self.config,
            tuple(self.registry),
            self.provider,
            self.runtime_context_provider,
        )
        effective_mode = mode or snapshot.config.execution_mode
        self._mode.set_mode(effective_mode)
        active_registry = self._mode.filter_tools(snapshot.registry)
        self.cancellation_token.reset()

        context_messages = build_turn_context_messages(
            question, effective_mode, snapshot, self._resumed_notice
        )
        self._resumed_notice = None
        history_messages = context_messages + self.history_messages()
        turn_messages: list[AgentMessage] = [UserMessage(content=question)]

        self._agent = Agent(self._gate.adapt_tools(active_registry))

        def tools_for_mode_fn(
            reg: tuple[ToolSpec, ...], m: ExecutionMode
        ) -> list[AgentTool]:
            filtered = policy_for_mode(m).filter_tools(reg)
            return self._gate.adapt_tools(filtered)

        loop_config = build_loop_config(
            mode=effective_mode,
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
            tools_for_mode=tools_for_mode_fn,
            steer=self.steer,
            emit_hook=lambda rec: _emit_hook(self._hook_manager, rec),
            mode_state=self._mode,
            get_prompt_version=_get_prompt_version,
        )

        _emit_hook(
            self._hook_manager,
            HookRecord(
                "before_agent_start",
                metadata={"question": question, "mode": effective_mode},
            ),
        )

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

        result = self._agent.last_result
        assert result is not None

        self._history.save_turn(result.messages)
        self._last_prompt_tokens = record_last_prompt_tokens(result.messages)

        visible_result = (
            replace(result, messages=context_messages + result.messages)
            if context_messages
            else result
        )
        final = _build_structured_result(
            visible_result, snapshot.config.max_steps, self._mode.current_mode
        )

        yield _final_event(result.steps, final)

    # ── 内部 ──

    def _reset_provider_conversation_state(self) -> None:
        reset = getattr(self.provider, "reset_conversation_state", None)
        if callable(reset):
            reset()
