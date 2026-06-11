from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from xcode.ai.events import ToolCall
from ..config import ExecutionMode
from ..observability.permissions import PermissionDecision
from ..skills import ToolSpec

"""Plan / Review / Act 的工具可见性策略。"""


class ExecutionPolicy(Protocol):
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]: ...

    def check_call(self, call: ToolCall) -> PermissionDecision: ...


PLAN_TOOL_NAMES = {
    "read_file",
    "glob_files",
    "grep_search",
}

REVIEW_EXTRA_TOOL_NAMES = {
    "bash",
}

REVIEW_BASH_PREFIXES = (
    "ls",
    "dir",
    "find",
    "rg",
    "grep",
    "git status",
    "git diff",
    "git show --stat",
    "git show --oneline",
)


@dataclass(frozen=True)
class ReviewCommand:
    text: str

    @classmethod
    def from_tool_input(cls, action_input: dict[str, object]) -> "ReviewCommand":
        command = action_input.get("command")
        if command is None:
            return cls(text="")
        return cls(text=str(command).lower().strip())

    def is_read_only_inspection(self) -> bool:
        return any(
            self.text == prefix or self.text.startswith(prefix + " ")
            for prefix in REVIEW_BASH_PREFIXES
        )


class PlanPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tuple(tool for tool in tools if tool.name in PLAN_TOOL_NAMES)

    def check_call(self, call: ToolCall) -> PermissionDecision:
        return "allow" if call.name in PLAN_TOOL_NAMES else "deny"


class ReviewPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        allowed = PLAN_TOOL_NAMES | REVIEW_EXTRA_TOOL_NAMES
        return tuple(tool for tool in tools if tool.name in allowed)

    def check_call(self, call: ToolCall) -> PermissionDecision:
        if call.name in PLAN_TOOL_NAMES:
            return "allow"
        if call.name == "bash":
            return "allow" if _is_review_bash_allowed(call.input) else "ask"
        return "deny"


class ActPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tools

    def check_call(self, call: ToolCall) -> PermissionDecision:
        return "allow"


def parse_execution_mode(value: object) -> ExecutionMode | None:
    if not isinstance(value, str):
        return None
    match value:
        case "plan":
            return "plan"
        case "review":
            return "review"
        case "act":
            return "act"
        case _:
            return None


def policy_for_mode(mode: ExecutionMode) -> ExecutionPolicy:
    if mode == "plan":
        return PlanPolicy()
    if mode == "review":
        return ReviewPolicy()
    return ActPolicy()


def registry_for_mode(
    registry: tuple[ToolSpec, ...],
    mode: ExecutionMode,
) -> tuple[ToolSpec, ...]:
    return policy_for_mode(mode).filter_tools(registry)


def mode_notice(mode: ExecutionMode) -> str:
    if mode == "plan":
        return (
            '<execution-mode name="plan">\n'
            "Plan Mode is active. Tools are limited to read/list/grep and static inspection. "
            "Do not modify files or run shell commands. Return only a concise plan.\n"
            "</execution-mode>"
        )
    if mode == "review":
        return (
            '<execution-mode name="review">\n'
            "Review Mode is active. Tools are read-only by default. Edits are unavailable; "
            "validation and static checks require approval. Report findings and risks only.\n"
            "</execution-mode>"
        )
    return ""


def _is_review_bash_allowed(action_input: dict[str, object]) -> bool:
    return ReviewCommand.from_tool_input(action_input).is_read_only_inspection()
