"""CodingAgentHarness — 编码领域专用的 agent 运行时。

继承自 AgentHarness，添加执行模式（plan/build/act）和技能激活。
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
from xcode.agent.results import AgentLoopResult
from xcode.ai.providers.base import ModelProvider
from xcode.harness.config import AgentConfig
from xcode.harness.skill_activation import (
    ExplicitSkillActivationResult,
    is_skill_activation_content,
)
from xcode.harness.agent_runtime._mode_protocol import ToolGateMode
from xcode.harness.agent_runtime.agent_helpers import run_coro_sync
from xcode.harness.agent_runtime.config import (
    AgentRuntimeConfig,
    GateConfig,
    TurnSnapshot,
    build_turn_context_messages,
)
from xcode.harness.agent_runtime.events import AgentHarnessEvent
from xcode.harness.agent_runtime.goal import GoalController
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
    """编码 agent 运行时。添加执行模式和技能激活。"""

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
        self._session_history = runtime.session_history
        self._todo_state = runtime.todo_state
        self._goal_session_id = gate.session_id
        super().__init__(provider, registry, config, gate, runtime)
        self._goal = GoalController(lambda: self.provider)
        if self._session_history is not None:
            self._session_history.set_session_id(self.session_id)
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
            "completion_verifier": self._goal.completion_feedback,
        }

    def _build_result(self, visible_result: object) -> AgentHarnessResult:
        from xcode.harness.agent_runtime.result import _build_structured_result

        result = _build_structured_result(cast(AgentLoopResult, visible_result))
        goal_notice = self._goal.consume_terminal_notice()
        answer = result.answer
        if goal_notice:
            answer = f"{answer}\n\n[{goal_notice}]".strip()
        return replace(
            result,
            answer=answer,
            run_state=CodingRunState(
                messages=result.messages,
                current_mode=self._mode.current_mode,
                goal=self._goal.state,
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

    def load_run_state(self, run_state: RunState) -> None:
        super().load_run_state(run_state)
        self.restore_run_state_metadata(run_state)

    def restore_run_state_metadata(self, payload: object) -> None:
        """恢复模式与 todo，但不替换已经重建好的消息历史。"""
        run_state = (
            payload
            if isinstance(payload, CodingRunState)
            else CodingRunState.from_dict(payload)
        )
        self._mode.set_mode(run_state.current_mode)
        if run_state.goal is not None:
            self._goal.restore(run_state.goal)
        if self._todo_state is not None:
            self._todo_state.replace(run_state.todos or [])

    def clear_history(self) -> None:
        super().clear_history()
        self._goal.clear()
        if self._coding_runtime.skill_registry is not None:
            self._coding_runtime.skill_registry.clear_activations()
        if self._todo_state is not None:
            self._todo_state.replace([])

    # ── 编码特定公共 API ──

    def available_skill_names(self) -> tuple[str, ...]:
        """返回当前运行时允许显式激活的技能名称。"""
        registry = self._coding_runtime.skill_registry
        return registry.available_names() if registry is not None else ()

    def set_history_session_id(self, session_id: str) -> None:
        """绑定 history 工具当前读取的真实 session。"""
        if session_id != self._goal_session_id:
            self._goal.clear()
            self._goal_session_id = session_id
        if self._session_history is not None:
            self._session_history.set_session_id(session_id)

    def set_goal(self, condition: str) -> None:
        """设置当前 session 的自然语言停止条件。"""
        self._goal.set(condition)

    def clear_goal(self) -> None:
        """清除当前停止条件。"""
        self._goal.clear()

    def pause_goal(self) -> None:
        """暂停当前停止条件的验收与自动重入。"""
        self._goal.pause()

    def resume_goal(self) -> None:
        """恢复当前停止条件的验收与自动重入。"""
        self._goal.resume()

    @property
    def goal_state(self) -> dict[str, str | bool | int | None]:
        """返回可持久化的 Goal 状态。"""
        return self._goal.state.to_dict()

    def restore_goal_state(self, payload: object) -> None:
        """从 session 事件恢复 Goal 状态。"""
        from xcode.harness.agent_runtime.goal import GoalState

        self._goal.restore(GoalState.from_dict(payload))

    @property
    def goal_condition(self) -> str | None:
        """返回当前停止条件。"""
        return self._goal.condition

    @property
    def goal_paused(self) -> bool:
        """返回当前停止条件是否暂停。"""
        return self._goal.paused

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
