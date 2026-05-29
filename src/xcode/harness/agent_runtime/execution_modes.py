from __future__ import annotations

import json
from typing import Literal, Protocol

from .events import ToolCall
from ..config import ExecutionMode
from ..skills import ToolSpec

"""Plan / Review / Act 的工具可见性策略。"""

ExecutionDecision = Literal["allow", "deny", "require_approval"]


class ExecutionPolicy(Protocol):
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]: ...

    def check_call(self, call: ToolCall) -> ExecutionDecision: ...


PLAN_TOOL_NAMES = {
    "read_file",
    "glob_files",
    "grep_search",
}

REVIEW_EXTRA_TOOL_NAMES = {
    "bash",
    "run_validation",
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


class PlanPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tuple(tool for tool in tools if tool.name in PLAN_TOOL_NAMES)

    def check_call(self, call: ToolCall) -> ExecutionDecision:
        return "allow" if call.name in PLAN_TOOL_NAMES else "deny"


class ReviewPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        allowed = PLAN_TOOL_NAMES | REVIEW_EXTRA_TOOL_NAMES
        return tuple(tool for tool in tools if tool.name in allowed)

    def check_call(self, call: ToolCall) -> ExecutionDecision:
        if call.name in PLAN_TOOL_NAMES:
            return "allow"
        if call.name == "run_validation":
            return "require_approval"
        if call.name == "bash":
            return (
                "allow"
                if _is_review_bash_allowed(_stringify_call_input(call.input))
                else "require_approval"
            )
        return "deny"


class ActPolicy:
    def filter_tools(self, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        return tools

    def check_call(self, call: ToolCall) -> ExecutionDecision:
        if call.name == "run_validation":
            return "require_approval"
        return "allow"


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


def _is_review_bash_allowed(action_input: str) -> bool:
    command = _command_from_tool_input(action_input).lower().strip()
    return any(
        command == prefix or command.startswith(prefix + " ")
        for prefix in REVIEW_BASH_PREFIXES
    )


def _command_from_tool_input(action_input: str) -> str:
    text = action_input.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if "command" in data:
            return str(data["command"])
        nested = data.get("input")
        if isinstance(nested, dict) and "command" in nested:
            return str(nested["command"])
        if nested is not None:
            return str(nested)
    return text


def _stringify_call_input(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
