"""执行模式管理（plan / review / act）。

负责模式状态切换、计划超时、模式工具构建、工具可见性过滤。
"""

from __future__ import annotations

from ..config import ExecutionMode
from ..skills import ToolSpec
from .execution_modes import policy_for_mode


class ModeManager:
    """管理 plan / review / act 模式切换和相关状态。"""

    MAX_PLAN_TURNS = 8

    def __init__(self, max_plan_turns: int = 8) -> None:
        self._current_mode: ExecutionMode = "act"
        self._plan_enter_step: int = 0
        self._max_plan_turns = max_plan_turns

    @property
    def current_mode(self) -> ExecutionMode:
        return self._current_mode

    def check_plan_timeout(self) -> bool:
        """检查 plan 模式是否超时，超时则自动切回 act。

        返回 True 表示发生了超时切换。
        """
        if self._current_mode != "plan":
            return False
        self._plan_enter_step += 1
        if self._plan_enter_step >= self._max_plan_turns:
            self._plan_enter_step = 0
            self._current_mode = "act"
            return True
        return False

    def filter_tools_for_mode(
        self, registry: tuple[ToolSpec, ...]
    ) -> tuple[ToolSpec, ...]:
        """根据当前模式过滤工具集。"""
        policy = policy_for_mode(self._current_mode)
        return policy.filter_tools(registry)
