"""StructuredAgent — harness 层对 agent/Agent 的适配。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from uuid import uuid4

from xcode.ai.providers.protocol import StreamProvider

from ...agent.agent import Agent
from ...agent.config import AgentContext, BeforeToolCallContext
from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from ...agent.protocols import AgentTool, AgentToolResult
from ...agent.types import TextContent, ToolCallContent
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
from ..observability import HookRecord, RuntimeCorrelation
from ..observability.permission_model import GrantStore
from ..skill_activation import (
    ExplicitSkillActivationResult,
    is_skill_activation_content,
)
from ..skills import ApprovalCallback, ToolRegistryState, ToolSpec

_PROMPT_VERSION_CACHE: str | None = None


def _get_prompt_version() -> str:
    global _PROMPT_VERSION_CACHE
    if _PROMPT_VERSION_CACHE is None:
        from .prompting.identity import PROMPT_VERSION as _v

        _PROMPT_VERSION_CACHE = _v
    return _PROMPT_VERSION_CACHE or "unknown"


__all__ = ["StructuredAgent"]


class StructuredAgent:
    def __init__(
        self,
        provider: StreamProvider,
        registry: tuple[ToolSpec, ...] | ToolRegistryState,
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
        self._runtime = runtime
        self._registry_state = (
            registry
            if isinstance(registry, ToolRegistryState)
            else ToolRegistryState(registry)
        )
        self.config = config
        self.compactor = runtime.compactor
        self._compact_controller = runtime.compact_controller
        self.runtime_context_provider = runtime.runtime_context_provider
        self.cancellation_token = runtime.cancellation_token or CancellationToken()
        self.request_hygiene = runtime.request_hygiene or RequestHygieneConfig()
        self._todo_state = runtime.todo_state
        self._correlation = gate.correlation or RuntimeCorrelation(gate.session_id)
        self._last_prompt_tokens: int | None = None

        self._hook_manager = gate.hook_manager
        self._mode = ExecutionModeState()
        resolved_permission_policy = resolve_permission_policy(
            runtime.project_root, gate.permission_policy
        )
        self.permission_policy = resolved_permission_policy
        self.restricted_dirs = gate.restricted_dirs
        self.hook_constraint_providers = gate.hook_constraint_providers
        self._gate = ToolGate(
            mode_state=self._mode,
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
        )
        self.audit_logger = gate.audit_logger
        self._history = HistoryManager()
        self._resumed_notice: str | None = None

    # ── 公共 API ──

    @property
    def registry(self) -> tuple[ToolSpec, ...]:
        """返回当前工具注册表快照。"""
        return self._registry_state.snapshot()

    @property
    def tool_map(self) -> dict[str, ToolSpec]:
        """按名称返回当前工具映射。"""
        return {tool.name: tool for tool in self.registry}

    def steer(self, msg: AgentMessage) -> None:
        self._agent.steer(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        self._agent.follow_up(msg)

    def request_compaction(self) -> None:
        if self._compact_controller is not None:
            self._compact_controller.request()

    def clear_history(self) -> None:
        self._history.clear()
        if self._todo_state is not None:
            self._todo_state.replace([])
        if self._runtime.skill_registry is not None:
            self._runtime.skill_registry.clear_activations()
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

    def load_history(self, messages: list[AgentMessage]) -> None:
        self._history.load(messages)
        if self._runtime.skill_registry is not None:
            self._runtime.skill_registry.restore_activations(messages)
        self._reset_provider_conversation_state()

    def set_resumed_notice(self, notice: str) -> None:
        self._resumed_notice = notice

    def load_run_state(self, run_state: RunState) -> None:
        self._history.load_run_state(run_state)
        if self._todo_state is not None:
            self._todo_state.replace(
                [
                    {
                        "id": item.id,
                        "content": item.content,
                        "status": item.status,
                    }
                    for item in run_state.todos
                ]
            )
        if self._runtime.skill_registry is not None:
            self._runtime.skill_registry.restore_activations(run_state.messages)
        self._reset_provider_conversation_state()
        restored = self._history.restore_mode(run_state)
        if restored is not None:
            self._mode.set_mode(restored)

    def history_messages(self) -> list[AgentMessage]:
        return self._history.messages()

    def available_skill_names(self) -> tuple[str, ...]:
        """返回当前运行时允许显式激活的技能名称。"""
        registry = self._runtime.skill_registry
        return registry.available_names() if registry is not None else ()

    def activate_skill(self, skill_name: str) -> ExplicitSkillActivationResult:
        """通过 canonical load_skill 工具显式激活技能。"""
        name = skill_name.strip()
        unavailable = self._explicit_skill_unavailable(name)
        if unavailable is not None:
            return unavailable
        load_skill = next(tool for tool in self.registry if tool.name == "load_skill")

        tool_call_id = f"explicit-skill-{uuid4().hex}"
        execution = self._execute_explicit_skill(
            name,
            load_skill,
            tool_call_id,
        )
        if isinstance(execution, ExplicitSkillActivationResult):
            return execution
        assistant_message, tool_result = execution
        return self._record_explicit_skill_activation(
            name,
            tool_call_id,
            load_skill.name,
            assistant_message,
            tool_result,
        )

    def _explicit_skill_unavailable(
        self,
        name: str,
    ) -> ExplicitSkillActivationResult | None:
        """返回显式激活前的名称或运行时可用性错误。"""
        if not name:
            return ExplicitSkillActivationResult(
                name="",
                status="unknown",
                message="Skill name is required.",
            )
        skill_registry = self._runtime.skill_registry
        if skill_registry is None:
            return ExplicitSkillActivationResult(
                name=name,
                status="disabled",
                message="Skills are disabled for this runtime.",
            )
        if not skill_registry.contains(name):
            return ExplicitSkillActivationResult(
                name=name,
                status="unknown",
                message=f"Unknown skill: {name}",
            )
        if not skill_registry.is_available(name):
            return ExplicitSkillActivationResult(
                name=name,
                status="disabled",
                message=f"Skill is unavailable for explicit activation: {name}",
            )
        if not any(tool.name == "load_skill" for tool in self.registry):
            return ExplicitSkillActivationResult(
                name=name,
                status="disabled",
                message="The skills tool group is disabled.",
            )
        return None

    def _execute_explicit_skill(
        self,
        name: str,
        load_skill: ToolSpec,
        tool_call_id: str,
    ) -> tuple[AssistantMessage, AgentToolResult] | ExplicitSkillActivationResult:
        """通过 ToolGate 执行单次显式 load_skill 调用。"""
        arguments: dict[str, object] = {"name": name}
        tool_call = ToolCallContent(
            id=tool_call_id,
            name=load_skill.name,
            arguments=arguments,
        )
        assistant_message = AssistantMessage(content=[tool_call])
        before_hook = self._gate.build_before_tool_hook(
            self._gate.snapshot_for((load_skill,))
        )
        before_result = before_hook(
            BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=arguments,
                context=AgentContext(),
            ),
            None,
        )
        if before_result is not None and before_result.block:
            return ExplicitSkillActivationResult(
                name=name,
                status="blocked",
                message=before_result.reason or f"Skill activation blocked: {name}",
            )
        if before_result is not None and before_result.args is not None:
            arguments = before_result.args

        try:
            adapted_tool = self._gate.adapt_tools((load_skill,))[0]
            tool_result = run_coro_sync(
                adapted_tool.execute(tool_call_id, arguments, None)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ExplicitSkillActivationResult(
                name=name,
                status="error",
                message=f"Failed to activate skill {name}: {exc}",
            )
        return assistant_message, tool_result

    def _record_explicit_skill_activation(
        self,
        name: str,
        tool_call_id: str,
        tool_name: str,
        assistant_message: AssistantMessage,
        tool_result: AgentToolResult,
    ) -> ExplicitSkillActivationResult:
        """分类激活结果，并仅记录首次成功的 canonical 消息对。"""
        content = "".join(
            block.text
            for block in tool_result.content
            if isinstance(block, TextContent)
        )
        if tool_result.is_error:
            return ExplicitSkillActivationResult(
                name=name,
                status="error",
                message=content or f"Failed to activate skill: {name}",
            )
        if 'status="already-active"' in content:
            return ExplicitSkillActivationResult(
                name=name,
                status="already_active",
                message=f"Skill already active: {name}",
                content=content,
            )
        if not is_skill_activation_content(content):
            return ExplicitSkillActivationResult(
                name=name,
                status="error",
                message=content or f"Skill activation returned no content: {name}",
            )

        result_message = ToolResultMessage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=content,
        )
        self._history.save_turn([assistant_message, result_message])
        self._reset_provider_conversation_state()
        return ExplicitSkillActivationResult(
            name=name,
            status="activated",
            message=f"Activated skill: {name}",
            content=content,
            tool_call_id=tool_call_id,
        )

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
        self._correlation.reset(self.session_id)

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
            project_root=self.project_root,
            skill_registry=self._runtime.skill_registry,
            prompt_instructions=self._runtime.prompt_instructions,
            correlation=self._correlation,
        )

        current = self._correlation.snapshot()
        _emit_hook(
            self._hook_manager,
            HookRecord(
                "before_agent_start",
                metadata={"question": question, "mode": effective_mode},
                timestamp=current.timestamp,
                session_id=current.session_id,
                turn_id=current.turn_id,
                request_id=current.request_id,
            ),
        )

        translation_state = _StreamTranslationState(correlation=self._correlation)

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
            visible_result,
            snapshot.config.max_steps,
            self._mode.current_mode,
            self._todo_state.snapshot() if self._todo_state is not None else (),
        )

        yield _final_event(
            result.steps,
            final,
            self._correlation.snapshot(),
        )

    # ── 内部 ──

    def _reset_provider_conversation_state(self) -> None:
        reset = getattr(self.provider, "reset_conversation_state", None)
        if callable(reset):
            reset()
