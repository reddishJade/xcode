from __future__ import annotations

from .repl_tools import brief_input
from xcode.harness.observability import (
    HITLResult,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolInput, ToolSpec, stringify_tool_input


class ReplHITLHandler:
    def __init__(
        self,
        session_policy: SessionPermissionPolicy,
        persistent_store: PersistentPermissionStore,
        prompt: object | None = None,
    ) -> None:
        self.session_policy = session_policy
        self.persistent_store = persistent_store
        self._prompt = prompt

    def __call__(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        action_input_text = stringify_tool_input(action_input)
        session_decision = self.session_policy.decide(tool.name, action_input_text)
        if session_decision is not None and session_decision != "ask":
            return HITLResult(session_decision, "session")
        persistent_policy = self.persistent_store.load()
        pers_decision = persistent_policy.decide(tool.name, action_input_text)
        if pers_decision is not None and pers_decision != "ask":
            return HITLResult(pers_decision, "permanent")
        return self._interactive_prompt(tool, action_input)

    def _interactive_prompt(
        self, tool: ToolSpec, action_input: ToolInput
    ) -> HITLResult:
        import questionary

        brief = brief_input(tool.name, action_input)
        choice = questionary.select(
            f"需要授权：{tool.name}  风险：{tool.risk}\n指令：{brief}",
            choices=[
                "允许（仅本次）",
                "此次对话中允许",
                "始终允许",
                "拒绝",
            ],
        ).ask()
        return self._apply_choice(choice, tool, action_input)

    def _apply_choice(
        self, choice: str | None, tool: ToolSpec, action_input: ToolInput
    ) -> HITLResult:
        action_input_text = stringify_tool_input(action_input)
        if choice == "允许（仅本次）":
            return HITLResult("allow", "once")
        if choice == "此次对话中允许":
            self.session_policy.grant(tool.name, "allow", action_input_text)
            return HITLResult("allow", "session")
        if choice == "始终允许":
            self.persistent_store.grant(tool.name, "allow", action_input_text)
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once")
