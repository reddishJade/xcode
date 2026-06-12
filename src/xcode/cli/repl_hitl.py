from __future__ import annotations

import asyncio
from queue import Queue
import shlex
from threading import Thread

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
        choice = _ask_hitl_choice(tool, action_input)
        return self._apply_choice(choice, tool, action_input)

    def _apply_choice(
        self, choice: str | None, tool: ToolSpec, action_input: ToolInput
    ) -> HITLResult:
        action_input_text = stringify_tool_input(action_input)
        input_prefix = _permission_input_prefix(tool.name, action_input)
        if choice == "允许（仅本次）":
            return HITLResult("allow", "once")
        if choice == "此次对话中允许":
            self.session_policy.grant(
                tool.name,
                "allow",
                None if input_prefix else action_input_text,
                input_prefix=input_prefix,
            )
            return HITLResult("allow", "session")
        if choice == "始终允许":
            self.persistent_store.grant(
                tool.name,
                "allow",
                None if input_prefix else action_input_text,
                input_prefix=input_prefix,
            )
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once")


def _ask_hitl_choice(tool: ToolSpec, action_input: ToolInput) -> str | None:
    """在同步或异步调用栈中安全显示 HITL 提示。"""
    if _has_running_event_loop():
        return _ask_hitl_choice_in_thread(tool, action_input)
    return _ask_hitl_choice_directly(tool, action_input)


def _ask_hitl_choice_directly(tool: ToolSpec, action_input: ToolInput) -> str | None:
    """直接显示授权选择，适用于当前线程没有事件循环的场景。"""
    import questionary

    brief = brief_input(tool.name, action_input)
    return questionary.select(
        f"需要授权：{tool.name}  风险：{tool.risk}\n指令：{brief}",
        choices=[
            "允许（仅本次）",
            "此次对话中允许",
            "始终允许",
            "拒绝",
        ],
    ).ask()


def _ask_hitl_choice_in_thread(tool: ToolSpec, action_input: ToolInput) -> str | None:
    """把阻塞式交互放到无事件循环的线程，避免影响当前循环。"""
    results: Queue[str | None | BaseException] = Queue(maxsize=1)

    def run_prompt() -> None:
        try:
            results.put(_ask_hitl_choice_directly(tool, action_input))
        except BaseException as exc:
            results.put(exc)

    thread = Thread(target=run_prompt, name="xcode-hitl-prompt", daemon=True)
    thread.start()
    result = results.get()
    thread.join()
    if isinstance(result, BaseException):
        raise result
    return result


def _has_running_event_loop() -> bool:
    """判断当前线程是否已经处于 asyncio event loop 内。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _permission_input_prefix(tool_name: str, action_input: ToolInput) -> str | None:
    """为可泛化的工具输入生成权限前缀。"""
    if tool_name != "bash":
        return None
    command = action_input.get("command") or action_input.get("input")
    if not isinstance(command, str):
        return None
    command_prefix = _bash_command_family(command)
    if command_prefix is None:
        return None
    return stringify_tool_input({"command": command_prefix})[:-2]


def _bash_command_family(command: str) -> str | None:
    """把常见验证命令归一到可复用的命令族。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if tokens[:3] == ["uv", "run", "pyright"]:
        return "uv run pyright"
    if tokens[:4] == ["uv", "run", "ruff", "check"]:
        return "uv run ruff check"
    if tokens[:4] == ["uv", "run", "ruff", "format"]:
        return "uv run ruff format"
    if tokens[:5] == ["uv", "run", "python", "-m", "unittest"]:
        return "uv run python -m unittest"
    return None
