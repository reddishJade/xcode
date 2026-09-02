from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from xcode.agent.messages import AgentMessage, UserMessage
from xcode.agent.types import ApprovalCallback, ToolSpec
from xcode.coding_agent.execution_modes import ExecutionMode
from xcode.harness.agent_runtime import (
    AgentHarnessEvent,
    BusyMessageMode,
    CancellationToken,
    SubmitOutcome,
)
from xcode.harness.observability import ExternalHookDiagnostic
from xcode.harness.session import SessionStore
from xcode.harness.skill_activation import ExplicitSkillActivationResult


class ToolRegistryApp(Protocol):
    @property
    def registry(self) -> tuple[ToolSpec, ...]: ...


class ReplAgent(Protocol):
    @property
    def current_approval_callback(self) -> ApprovalCallback | None: ...

    @property
    def user_approval_callback(self) -> ApprovalCallback | None: ...

    @user_approval_callback.setter
    def user_approval_callback(self, value: ApprovalCallback | None) -> None: ...

    @property
    def auto_approval_callback(self) -> ApprovalCallback | None: ...

    cancellation_token: CancellationToken

    def steer(
        self,
        msg: UserMessage,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome: ...

    def followup(
        self,
        msg: UserMessage,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome: ...

    def inject(self, msg: AgentMessage) -> SubmitOutcome: ...

    def submit_busy_message(
        self,
        msg: UserMessage,
        mode: BusyMessageMode = BusyMessageMode.STEER,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome: ...

    def interrupt(self, reason: str = "interrupted by user") -> bool: ...

    def has_pending_input(self) -> bool: ...

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def restore_run_state_metadata(self, payload: object) -> None: ...

    def request_context_window(self) -> bool: ...

    def set_goal(self, condition: str) -> None: ...

    def clear_goal(self) -> None: ...

    def pause_goal(self) -> None: ...

    def resume_goal(self) -> None: ...

    @property
    def goal_state(self) -> dict[str, str | bool | int | None]: ...

    def restore_goal_state(self, payload: object) -> None: ...

    @property
    def goal_condition(self) -> str | None: ...

    @property
    def goal_paused(self) -> bool: ...

    def available_skill_names(self) -> tuple[str, ...]: ...

    def activate_skill(
        self, skill_name: str, mode: ExecutionMode | None = None
    ) -> ExplicitSkillActivationResult: ...


class ModelControlApp(Protocol):
    def get_model_info(self) -> dict[str, str]: ...

    def set_model(
        self,
        *,
        model: str,
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str: ...


class ReplApp(ModelControlApp, ToolRegistryApp, Protocol):
    @property
    def agent(self) -> ReplAgent: ...

    @property
    def registry(self) -> tuple[ToolSpec, ...]: ...

    @property
    def session_store(self) -> SessionStore: ...

    def ask_stream(
        self,
        question: str | None,
        mode: ExecutionMode | None = None,
        *,
        display_question: str | None = None,
    ) -> Iterator[AgentHarnessEvent]: ...

    def restore_session(self) -> None: ...

    def hook_diagnostics(self) -> tuple[ExternalHookDiagnostic, ...]: ...

    def record_context_window_reset(
        self,
        *,
        window_id: str,
        messages_before: int,
        messages_after: int,
        replacement: list[AgentMessage],
    ) -> str: ...
