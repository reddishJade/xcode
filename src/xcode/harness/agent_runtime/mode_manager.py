"""执行模式管理（plan / review / act）。

负责模式状态切换、计划超时、模式工具构建、工具可见性过滤。
"""

from __future__ import annotations

from typing import Any

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
        self._plan_pending_confirmation: bool = False

    @property
    def current_mode(self) -> ExecutionMode:
        return self._current_mode

    @property
    def plan_pending_confirmation(self) -> bool:
        return self._plan_pending_confirmation

    def switch_to_plan(self) -> str:
        """切换到 plan 模式，返回提示消息。"""
        self._current_mode = "plan"
        self._plan_enter_step = 0
        return (
            "Entered Plan Mode. Tools are limited to read-only. "
            "Investigate and report a plan. "
            "Use exit_plan_mode with a concise plan summary to return to full tool access."
        )

    def switch_to_act(self, plan_summary: str | None = None) -> tuple[str, str | None]:
        """切换到 act 模式，返回 (提示消息, plan_summary)。

        调用方负责 steer plan_summary 到 agent 并更新工具列表。
        """
        self._current_mode = "act"
        self._plan_pending_confirmation = True
        return "Plan ready. Present it to the user for confirmation.", plan_summary

    def is_plan_confirmation_required(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> bool:
        """检查当前工具调用是否需要计划确认。"""
        if not self._plan_pending_confirmation:
            return False
        from xcode.ai.events import ToolCall

        policy = policy_for_mode("plan")
        decision = policy.check_call(ToolCall(id="", name=tool_name, input=tool_args))
        return decision != "allow"

    def confirm_plan(self) -> None:
        """确认计划，允许执行写工具。"""
        self._plan_pending_confirmation = False

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

    def build_mode_switch_tools(self) -> tuple[ToolSpec, ToolSpec]:
        """构建 plan/act 模式切换工具。"""
        plan_tool = ToolSpec(
            name="enter_plan_mode",
            description=(
                "Switch to Plan Mode: read-only tools only. "
                "Call this before making changes to investigate first."
            ),
            input_hint="empty",
            handler=lambda _input: self.switch_to_plan(),
            risk="low",
        )
        act_tool = ToolSpec(
            name="exit_plan_mode",
            description=(
                "Exit Plan Mode: return to full tool access. "
                "Call this with a concise summary of your plan."
            ),
            input_hint="plan_summary",
            handler=lambda _input: self.switch_to_act(
                _input.get("plan_summary", "")
                if isinstance(_input, dict)
                else str(_input)
            )[0],
            risk="low",
        )
        return plan_tool, act_tool

    def filter_tools_for_mode(
        self, registry: tuple[ToolSpec, ...]
    ) -> tuple[ToolSpec, ...]:
        """根据当前模式过滤工具集。"""
        policy = policy_for_mode(self._current_mode)
        return policy.filter_tools(registry)
