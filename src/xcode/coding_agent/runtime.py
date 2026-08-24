"""编码 agent 对通用运行时的扩展配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.memory import MemoryManager
from xcode.harness.session import SessionHistory
from xcode.harness.session_todo import SessionTodoState
from xcode.harness.skills import SkillRegistry

from .execution_modes import ExecutionMode


@dataclass(frozen=True)
class CodingAgentRuntimeConfig(AgentRuntimeConfig):
    """编码技能、记忆和任务状态的运行时配置。"""

    initial_mode: ExecutionMode = "act"
    approval_router: Literal["mode", "user", "auto"] = "mode"
    skill_registry: SkillRegistry | None = None
    memory_manager: MemoryManager | None = None
    session_history: SessionHistory | None = None
    todo_state: SessionTodoState | None = None
    prompt_instructions: tuple[dict, ...] = ()
