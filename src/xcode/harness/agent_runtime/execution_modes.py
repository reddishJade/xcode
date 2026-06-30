from __future__ import annotations

from typing import Protocol

from xcode.ai.events import ToolCall
from ..config import ExecutionMode
from ..observability.permissions import PermissionDecision
from ..skills import ToolSpec

"""Plan / Build / Act 的工具可见性策略。"""


class ExecutionPolicy(Protocol):
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]: ...

    def check_call(self, call: ToolCall) -> PermissionDecision: ...


class ExecutionModeState:
    """管理当前执行模式和 plan 模式超时状态。"""

    def __init__(self, max_plan_turns: int = 8) -> None:
        self._current_mode: ExecutionMode = "act"
        self._plan_enter_step = 0
        self._max_plan_turns = max_plan_turns

    @property
    def current_mode(self) -> ExecutionMode:
        return self._current_mode

    def set_mode(self, mode: ExecutionMode) -> None:
        """设置当前执行模式。"""
        self._current_mode = mode
        if mode == "plan":
            self._plan_enter_step = 0

    def check_plan_timeout(self) -> bool:
        """检查 plan 模式是否超时，超时则自动切换到 build。"""
        if self._current_mode != "plan":
            return False
        self._plan_enter_step += 1
        if self._plan_enter_step < self._max_plan_turns:
            return False
        self._plan_enter_step = 0
        self._current_mode = "build"
        return True

    def filter_tools(self, registry: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        """根据当前模式过滤工具集。"""
        return registry_for_mode(registry, self._current_mode)


PLAN_TOOL_NAMES = {
    "read_file",
    "glob_files",
    "grep_search",
    "find_files",
    "list_dir",
    "search_tools",
    "load_skill",
}

BUILD_TOOL_NAMES = {
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "glob_files",
    "grep_search",
    "find_files",
    "list_dir",
    "bash",
    "shell",
    "search_tools",
    "load_skill",
}


class PlanPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tuple(tool for tool in tools if tool.name in PLAN_TOOL_NAMES)

    def check_call(self, call: ToolCall) -> PermissionDecision:
        return "allow" if call.name in PLAN_TOOL_NAMES else "deny"


class BuildPolicy:
    """Build mode: ordinary file mutations allowed; high-risk through PermissionPipeline."""

    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tuple(tool for tool in tools if tool.name in BUILD_TOOL_NAMES)

    def check_call(self, call: ToolCall) -> PermissionDecision:
        if call.name in BUILD_TOOL_NAMES:
            return "allow"
        return "deny"


class ActPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tools

    def check_call(self, call: ToolCall) -> PermissionDecision:
        return "allow"


_POLICIES: dict[ExecutionMode, ExecutionPolicy] = {
    "plan": PlanPolicy(),
    "build": BuildPolicy(),
    "act": ActPolicy(),
}


def parse_execution_mode(value: object) -> ExecutionMode | None:
    if not isinstance(value, str):
        return None
    match value:
        case "plan":
            return "plan"
        case "build":
            return "build"
        case "act":
            return "act"
        case _:
            return None


def policy_for_mode(mode: ExecutionMode) -> ExecutionPolicy:
    return _POLICIES[mode]


def registry_for_mode(
    registry: tuple[ToolSpec, ...],
    mode: ExecutionMode,
) -> tuple[ToolSpec, ...]:
    return policy_for_mode(mode).filter_tools(registry)


def mode_notice(mode: ExecutionMode) -> str:
    if mode == "plan":
        return (
            '<execution-mode name="plan">\n'
            "Plan Mode is active. Use read-only inspection tools only. "
            "Do not modify files or run shell commands. Return only a concise plan.\n"
            "</execution-mode>"
        )
    if mode == "build":
        return (
            '<execution-mode name="build">\n'
            "Build Mode is active. Ordinary file mutations are allowed. "
            "High-risk shell commands still require approval. "
            "Execute the approved plan step by step.\n"
            "</execution-mode>"
        )
    return ""
