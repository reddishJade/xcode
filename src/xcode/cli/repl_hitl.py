from __future__ import annotations

import asyncio
from typing import Any

from .commands import PromptLike
from .repl_tools import brief_input
from xcode.harness.observability import (
    HITLResult,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolInput, ToolSpec, stringify_tool_input

radiolist_dialog: Any
try:
    from prompt_toolkit.shortcuts.dialogs import radiolist_dialog
except ImportError:
    radiolist_dialog = None


class ReplHITLHandler:
    def __init__(
        self,
        session_policy: SessionPermissionPolicy,
        persistent_store: PersistentPermissionStore,
        prompt: PromptLike | None = None,
    ) -> None:
        self.session_policy = session_policy
        self.persistent_store = persistent_store
        self.prompt = prompt

    def __call__(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        action_input_text = stringify_tool_input(action_input)
        session_decision = self.session_policy.decide(tool.name, action_input_text)
        if session_decision is not None and session_decision != "ask":
            return HITLResult(session_decision, "session")
        persistent_policy = self.persistent_store.load()
        pers_decision = persistent_policy.decide(tool.name, action_input_text)
        if pers_decision is not None and pers_decision != "ask":
            return HITLResult(pers_decision, "permanent")
        if should_use_radiolist(self.prompt):
            choice = radiolist_prompt(tool, action_input)
        elif self.prompt is not None and not is_prompt_toolkit_prompt(self.prompt):
            choice = self.prompt.prompt(self._prompt_text(tool, action_input)).strip()
        else:
            choice = input(self._terminal_prompt_text(tool, action_input)).strip()
        return self._apply_choice(choice, tool, action_input)

    def _prompt_text(self, tool: ToolSpec, action_input: ToolInput) -> str:
        brief = brief_input(tool.name, action_input)
        return (
            f"需要授权：{tool.name}"
            f"\n  指令：{brief}"
            f"\n  风险：{tool.risk}"
            f"\n  选项："
            f"\n    1) 允许（仅本次）"
            f"\n    2) 此次对话中允许"
            f"\n    3) 始终允许"
            f"\n    4) 拒绝"
        )

    def _terminal_prompt_text(self, tool: ToolSpec, action_input: ToolInput) -> str:
        return f"\r\033[K\n{self._prompt_text(tool, action_input)}\napprove [1-4]> "

    def _apply_choice(
        self, choice: str, tool: ToolSpec, action_input: ToolInput
    ) -> HITLResult:
        action_input_text = stringify_tool_input(action_input)
        if choice == "1":
            return HITLResult("allow", "once")
        if choice == "2":
            self.session_policy.grant(tool.name, "allow", action_input_text)
            return HITLResult("allow", "session")
        if choice == "3":
            self.persistent_store.grant(tool.name, "allow", action_input_text)
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once")


def has_radiolist() -> bool:
    return radiolist_dialog is not None


def should_use_radiolist(prompt: PromptLike | None) -> bool:
    if is_async_loop_running():
        return False
    if prompt is not None and is_prompt_toolkit_prompt(prompt):
        return False
    return has_radiolist()


def is_async_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def is_prompt_toolkit_prompt(prompt: PromptLike | None) -> bool:
    if prompt is None:
        return False
    module = type(prompt).__module__
    return module.startswith("prompt_toolkit.")


def radiolist_prompt(tool: ToolSpec, action_input: ToolInput) -> str:
    if radiolist_dialog is None:
        return "4"

    brief = brief_input(tool.name, action_input)
    result = radiolist_dialog(
        title=f"需要授权：{tool.name}",
        text=f"指令：{brief}    风险：{tool.risk}",
        values=[
            ("1", "允许（仅本次）"),
            ("2", "此次对话中允许"),
            ("3", "始终允许"),
            ("4", "拒绝"),
        ],
        default=None,
    ).run()
    return result or "4"
