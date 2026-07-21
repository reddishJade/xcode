"""CodingAgentHarness — 编码领域专用的 agent 运行时。

继承自 AgentHarness，添加执行模式（plan/build/act）、技能激活和记忆反馈。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from typing import cast
from uuid import uuid4

from xcode.agent.config import AgentContext, BeforeToolCallContext
from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
)
from xcode.agent.types import AgentToolResult
from xcode.agent.types import TextContent, ToolCallContent
from xcode.agent.types import ToolSpec
from xcode.agent.results import AgentLoopResult, TerminationReason
from xcode.ai.providers.base import ModelProvider
from xcode.harness.config import AgentConfig
from xcode.harness.skill_activation import (
    ExplicitSkillActivationResult,
    is_skill_activation_content,
)
from xcode.harness.memory.manager import MemoryOutcome
from xcode.harness.agent_runtime._mode_protocol import ToolGateMode
from xcode.harness.agent_runtime.agent_helpers import run_coro_sync
from xcode.harness.agent_runtime.config import (
    AgentRuntimeConfig,
    GateConfig,
    TurnSnapshot,
    build_turn_context_messages,
)
from xcode.harness.agent_runtime.events import AgentHarnessEvent
from xcode.harness.agent_runtime.harness import AgentHarness
from xcode.harness.agent_runtime.result import AgentHarnessResult, RunState
from .execution_modes import (
    ExecutionMode,
    ExecutionModeState,
    mode_notice,
)
from .state import CodingRunState
from .runtime import CodingAgentRuntimeConfig

__all__ = ["CodingAgentHarness"]


class CodingAgentHarness(AgentHarness):
    """编码 agent 运行时。添加执行模式、技能激活、记忆反馈。"""

    def __init__(
        self,
        provider: ModelProvider,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig | None = None,
        gate: GateConfig | None = None,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        gate = gate or GateConfig()
        runtime = runtime or CodingAgentRuntimeConfig()
        if not isinstance(runtime, CodingAgentRuntimeConfig):
            raise TypeError("CodingAgentHarness requires CodingAgentRuntimeConfig")
        self._coding_runtime = runtime
        self._mode = ExecutionModeState()
        self._memory_manager = runtime.memory_manager
        self._todo_state = runtime.todo_state
        super().__init__(provider, registry, config, gate, runtime)
        from xcode.coding_agent.tools.subagent import bind_subagent_permission_gate

        bind_subagent_permission_gate(self._registry, self._gate)

    # ── AgentHarness 扩展点覆盖 ──

    def _build_gate_mode(self) -> ToolGateMode:
        return self._mode

    def _build_active_registry(
        self, registry: tuple[ToolSpec, ...]
    ) -> tuple[ToolSpec, ...]:
        return self._mode.filter_tools(registry)

    def _build_context_messages(
        self,
        question: str,
        snapshot: TurnSnapshot,
    ) -> list[AgentMessage]:
        memory_overview: str | None = None
        if self._resumed_notice is not None and self._memory_manager is not None:
            from xcode.harness.agent_runtime.prompting.builder import (
                render_memory_overview,
            )

            memory_overview = render_memory_overview(self._memory_manager)
        return build_turn_context_messages(
            question,
            snapshot,
            self._resumed_notice,
            mode_notice=mode_notice(self._mode.current_mode),
            memory_overview=memory_overview,
        )

    def _build_loop_config_extras(self) -> dict:
        return {
            "mode_state": self._mode,
        }

    def _build_result(
        self, visible_result: object, max_steps: int
    ) -> AgentHarnessResult:
        from xcode.harness.agent_runtime.result import _build_structured_result

        result = _build_structured_result(
            cast(AgentLoopResult, visible_result),
            max_steps,
        )
        return replace(
            result,
            run_state=CodingRunState(
                messages=result.messages,
                current_mode=self._mode.current_mode,
                todos=(
                    self._todo_state.to_dicts()
                    if self._todo_state is not None
                    else None
                ),
            ),
        )

    def _post_load_history(self, messages: list[AgentMessage]) -> None:
        if self._coding_runtime.skill_registry is not None:
            self._coding_runtime.skill_registry.restore_activations(messages)

    def _post_run(self, final: AgentHarnessResult) -> None:
        self._record_memory_feedback(final)

    def load_run_state(self, run_state: RunState) -> None:
        super().load_run_state(run_state)
        if isinstance(run_state, CodingRunState):
            self._mode.set_mode(run_state.current_mode)
        if self._todo_state is not None:
            todos = run_state.todos if isinstance(run_state, CodingRunState) else []
            self._todo_state.replace(todos or [])

    def clear_history(self) -> None:
        super().clear_history()
        if self._coding_runtime.skill_registry is not None:
            self._coding_runtime.skill_registry.clear_activations()
        if self._todo_state is not None:
            self._todo_state.replace([])

    # ── 编码特定公共 API ──

    def available_skill_names(self) -> tuple[str, ...]:
        """返回当前运行时允许显式激活的技能名称。"""
        registry = self._coding_runtime.skill_registry
        return registry.available_names() if registry is not None else ()

    def activate_skill(
        self, skill_name: str, mode: ExecutionMode | None = None
    ) -> ExplicitSkillActivationResult:
        """通过 canonical load_skill 工具显式激活技能。"""
        name = skill_name.strip()
        unavailable = self._explicit_skill_unavailable(name)
        if unavailable is not None:
            return unavailable
        if mode is not None:
            self._mode.set_mode(mode)
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

    def run(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AgentHarnessResult:
        return run_coro_sync(self.arun(question, mode=mode))

    async def run_async(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AgentHarnessResult:
        return await self.arun(question, mode=mode)

    async def arun(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AgentHarnessResult:
        result: AgentHarnessResult | None = None
        async for event in self.arun_stream(question, mode=mode):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> Iterator[AgentHarnessEvent]:
        if mode is not None:
            self._mode.set_mode(mode)
        yield from super().run_stream(question)

    async def arun_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AsyncIterator[AgentHarnessEvent]:
        if mode is not None:
            self._mode.set_mode(mode)
        async for event in super().arun_stream(question):
            yield event

    # ── 技能激活内部 ──

    def _explicit_skill_unavailable(
        self,
        name: str,
    ) -> ExplicitSkillActivationResult | None:
        if not name:
            return ExplicitSkillActivationResult(
                name="",
                status="unknown",
                message="Skill name is required.",
            )
        skill_registry = self._coding_runtime.skill_registry
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
                message="Skill loading is disabled for this runtime.",
            )
        return None

    def _execute_explicit_skill(
        self,
        name: str,
        load_skill: ToolSpec,
        tool_call_id: str,
    ) -> tuple[AssistantMessage, AgentToolResult] | ExplicitSkillActivationResult:
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
        self._history.extend([assistant_message, result_message])
        self._reset_provider_conversation_state()
        return ExplicitSkillActivationResult(
            name=name,
            status="activated",
            message=f"Activated skill: {name}",
            content=content,
            tool_call_id=tool_call_id,
        )

    # ── 记忆反馈 ──

    def _record_memory_feedback(self, final: AgentHarnessResult) -> None:
        manager = self._memory_manager
        if manager is None:
            return
        if final.answer:
            manager.record_explicit_references(final.answer)
        outcome = self._memory_outcome_for_result(final)
        if outcome is None:
            return
        source = f"runtime:{self.session_id}"
        if outcome == "success":
            manager.adopt_injected_records(source=source)
        manager.record_session_outcome(outcome, source=source)

    def _memory_outcome_for_result(
        self, final: AgentHarnessResult
    ) -> MemoryOutcome | None:
        answer = final.answer.strip()
        if final.termination_reason is TerminationReason.COMPLETED:
            return "success" if answer else None
        if final.termination_reason in {
            TerminationReason.PROVIDER_ERROR,
            TerminationReason.WATCHDOG,
        }:
            return "failure"
        if final.termination_reason is TerminationReason.STEP_LIMIT and not answer:
            return "failure"
        return None
