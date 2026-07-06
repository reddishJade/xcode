"""Runtime-context binding without durable-memory prompt injection.

System prompt construction remains limited to host-owned runtime context. Durable
memory is carried as an explicit dependency for the structured ContextCollector
pipeline; it is never rendered by this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from xcode.coding_agent.tools.shell_adapter import ShellSpec
from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.session_todo import SessionTodoState
from xcode.harness.skills import ToolRegistryState, ToolSpec

from .builder import SystemPromptBuilder
from .builder import build_runtime_context_provider as _build_system_context_provider

if TYPE_CHECKING:
    from xcode.harness.memory import MemoryManager


@dataclass(frozen=True)
class RuntimeContextProvider:
    """Callable system-context provider with an explicit memory dependency.

    ``memory_manager`` is intentionally metadata for loop assembly, not an input
    to system-prompt rendering. The context collector owns memory retrieval and
    produces typed USER_CONTEXT blocks later in the request pipeline.
    """

    system_context: Callable[[str], list[str]]
    memory_manager: MemoryManager | None = None

    def __call__(self, question: str) -> list[str]:
        return self.system_context(question)


def build_runtime_context_provider(
    project_root: Path,
    registry: tuple[ToolSpec, ...] | ToolRegistryState,
    prompt_builder: SystemPromptBuilder | None = None,
    resumed_notice: Callable[[], str | None] | None = None,
    interrupted_notice: Callable[[], str | None] | None = None,
    contextual_state: ContextualRetrievalState | None = None,
    modules: tuple[str, ...] | None = None,
    shell_spec: ShellSpec | None = None,
    todo_state: SessionTodoState | None = None,
    memory_manager: MemoryManager | None = None,
) -> RuntimeContextProvider:
    """Build host-owned runtime context and bind memory for collector assembly."""
    system_context = _build_system_context_provider(
        project_root,
        registry,
        prompt_builder=prompt_builder,
        resumed_notice=resumed_notice,
        interrupted_notice=interrupted_notice,
        contextual_state=contextual_state,
        modules=modules,
        shell_spec=shell_spec,
        todo_state=todo_state,
    )
    return RuntimeContextProvider(
        system_context=system_context,
        memory_manager=memory_manager,
    )
