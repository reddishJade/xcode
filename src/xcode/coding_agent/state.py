"""编码 agent 的可持久化运行状态。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from xcode.harness.agent_runtime.result import RunState
from xcode.harness.agent_runtime.goal import GoalState

type ExecutionModeName = Literal["plan", "build", "act"]


@dataclass(frozen=True)
class CodingRunState(RunState):
    """包含编码执行模式与任务列表的运行状态。"""

    current_mode: ExecutionModeName = "act"
    last_agent: str = "main"
    needs_follow_up: bool = False
    todos: list[dict[str, Any]] | None = None
    goal: GoalState | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""
        return {
            **super().to_dict(),
            "current_mode": self.current_mode,
            "last_agent": self.last_agent,
            "needs_follow_up": self.needs_follow_up,
            "todos": self.todos or [],
            "goal": self.goal.to_dict() if self.goal is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: object) -> CodingRunState:
        """从 JSON 字典恢复编码运行状态。"""
        if not isinstance(payload, Mapping):
            return cls(messages=[])
        mode = payload.get("current_mode")
        current_mode: ExecutionModeName = (
            mode if mode in {"plan", "build", "act"} else "act"
        )
        raw_todos = payload.get("todos", [])
        todos = (
            [dict(item) for item in raw_todos if isinstance(item, Mapping)]
            if isinstance(raw_todos, list)
            else []
        )
        return cls(
            messages=RunState.from_dict(payload).messages,
            current_mode=current_mode,
            last_agent=str(payload.get("last_agent", "main")),
            needs_follow_up=bool(payload.get("needs_follow_up", False)),
            todos=todos,
            goal=(
                GoalState.from_dict(payload["goal"])
                if isinstance(payload.get("goal"), Mapping)
                else None
            ),
        )
