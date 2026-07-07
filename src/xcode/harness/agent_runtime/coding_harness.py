"""CodingAgentHarness — 编码领域专用的 agent 运行时。

继承自 AgentHarness，添加执行模式（plan/build/act）、技能激活和记忆反馈。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from copy import deepcopy
from uuid import uuid4

from ...agent.config import AgentContext, BeforeToolCallContext
from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
)
from ...agent.types import AgentTool, AgentToolResult
from ...agent.types import TextContent, ToolCallContent
from ...agent.types import ApprovalCallback, ToolSpec
from ...agent.results import TerminationReason
from xcode.ai.providers.base import ModelProvider
from ..config import AgentConfig, ExecutionMode
from ..skill_activation import (
    ExplicitSkillActivationResult,
    is_skill_activation_content,
)
from ..memory.manager import MemoryOutcome
from ._mode_protocol import ToolGateMode
from .agent_helpers import run_coro_sync
from .config import (
    AgentRuntimeConfig,
    GateConfig,
    TurnSnapshot,
    build_turn_context_messages,
)
from .events import CodingAgentHarnessEvent
from .harness import AgentHarness
from .result import RunState, CodingAgentHarnessResult
from xcode.coding_agent.execution_modes import (
    ExecutionModeState,
    policy_for_mode,
)

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
        runtime = runtime or AgentRuntimeConfig()
        self._mode = ExecutionModeState()
        self._memory_manager = runtime.memory_manager
        super().__init__(provider, registry, config, gate, runtime)

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
            from .prompting.builder import render_memory_overview

            memory_overview = render_memory_overview(self._memory_manager)
        return build_turn_context_messages(
            question,
            self._mode.current_mode,
            snapshot,
            self._resumed_notice,
            memory_overview=memory_overview,
        )

    def _build_loop_config_extras(self) -> dict:
        def tools_for_mode_fn(
            reg: tuple[ToolSpec, ...], m: ExecutionMode
        ) -> list[AgentTool]:
            filtered = policy_for_mode(m).filter_tools(reg)
            return self._gate.adapt_tools(filtered)

        return {
            "mode": self._mode.current_mode,
            "mode_state": self._mode,
            "tools_for_mode": tools_for_mode_fn,
            "skill_registry": self._runtime.skill_registry,
        }

    def _build_result(
        self, visible_result: object, max_steps: int
    ) -> CodingAgentHarnessResult:
        from .result import _build_structured_result

        return _build_structured_result(
            visible_result, max_steps, self._mode.current_mode  # type: ignore[arg-type]
        )

    def _post_load_history(self, messages: list[AgentMessage]) -> None:
        if self._runtime.skill_registry is not None:
            self._runtime.skill_registry.restore_activations(messages)

    def _post_run(self, final: CodingAgentHarnessResult) -> None:
        self._record_memory_feedback(final)

    def load_run_state(self, run_state: RunState) -> None:
        super().load_run_state(run_state)
        if run_state.current_mode in {"act", "plan", "build"}:
            self._mode.set_mode(run_state.current_mode)

    def clear_history(self) -> None:
        super().clear_history()
        if self._runtime.skill_registry is not None:
            self._runtime.skill_registry.clear_activations()

    # ── 编码特定公共 API ──

    def available_skill_names(self) -> tuple[str, ...]:
        """返回当前运行时允许显式激活的技能名称。"""
        registry = self._runtime.skill_registry
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
    ) -> CodingAgentHarnessResult:
        return run_coro_sync(self.arun(question, mode=mode))

    async def run_async(
        self, question: str, mode: ExecutionMode | None = None
    ) -> CodingAgentHarnessResult:
        return await self.arun(question, mode=mode)

    async def arun(
        self, question: str, mode: ExecutionMode | None = None
    ) -> CodingAgentHarnessResult:
        result: CodingAgentHarnessResult | None = None
        async for event in self.arun_stream(question, mode=mode):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> Iterator[CodingAgentHarnessEvent]:
        if mode is not None:
            self._mode.set_mode(mode)
        yield from super().run_stream(question)

    async def arun_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AsyncIterator[CodingAgentHarnessEvent]:
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

    def _record_memory_feedback(self, final: CodingAgentHarnessResult) -> None:
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
        self, final: CodingAgentHarnessResult
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
