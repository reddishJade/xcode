from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from xcode.agent.messages import AgentMessage
from xcode.harness.agent_runtime import CancellationToken, StructuredAgentEvent
from xcode.harness.config import ExecutionMode
from xcode.harness.skill_activation import ExplicitSkillActivationResult
from xcode.harness.skills import ApprovalCallback, ToolSpec


class ToolRegistryApp(Protocol):
    @property
    def registry(self) -> tuple[ToolSpec, ...]: ...


class ReplAgent(Protocol):
    @property
    def approval_callback(self) -> ApprovalCallback | None: ...
    @approval_callback.setter
    def approval_callback(self, value: ApprovalCallback | None) -> None: ...

    cancellation_token: CancellationToken

    def follow_up(self, msg: AgentMessage) -> None: ...

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def request_compaction(self) -> None: ...

    def available_skill_names(self) -> tuple[str, ...]: ...

    def activate_skill(self, skill_name: str) -> ExplicitSkillActivationResult: ...


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

    def ask_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> Iterator[StructuredAgentEvent]: ...
