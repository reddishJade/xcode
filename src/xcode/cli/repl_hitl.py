from __future__ import annotations

import asyncio
from queue import Queue
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
        if choice == "允许（仅本次）":
            return HITLResult("allow", "once")
        if choice == "此次对话中允许":
            self.session_policy.grant(tool.name, "allow", action_input_text)
            return HITLResult("allow", "session")
        if choice == "始终允许":
            self.persistent_store.grant(tool.name, "allow", action_input_text)
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once")


def _ask_hitl_choice(tool: ToolSpec, action_input: ToolInput) -> str | None:
    """在同步或异步调用栈中安全显示 HITL 提示。"""
    if _has_running_event_loop():
        return _ask_hitl_choice_in_thread(tool, action_input)
    return _ask_hitl_choice_directly(tool, action_input)


def _ask_hitl_choice_directly(tool: ToolSpec, action_input: ToolInput) -> str | None:
    """直接显示授权选择，适用于当前线程没有事件循环的场景。"""
    brief = brief_input(tool.name, action_input)
    return _run_hitl_dialog(
        title=f"需要授权：{tool.name}  风险：{tool.risk}",
        text=f"指令：{brief}",
        choices=_hitl_choices(),
    )


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


def _hitl_choices() -> list[tuple[str, str]]:
    """返回 HITL 选择项。"""
    return [
        ("允许（仅本次）", "允许（仅本次）"),
        ("此次对话中允许", "此次对话中允许"),
        ("始终允许", "始终允许"),
        ("拒绝", "拒绝"),
    ]


def _run_hitl_dialog(
    *,
    title: str,
    text: str,
    choices: list[tuple[str, str]],
) -> str | None:
    """显示支持方向键、鼠标和数字键选择的授权对话框。"""
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.layout.containers import HSplit
    from prompt_toolkit.shortcuts.dialogs import _create_app
    from prompt_toolkit.widgets import Button, Dialog, Label, RadioList

    radio_list = RadioList(
        values=choices,
        default="拒绝",
        show_numbers=True,
        select_on_focus=True,
    )

    def ok_handler() -> None:
        get_app().exit(result=radio_list.current_value)

    def cancel_handler() -> None:
        get_app().exit(result=None)

    dialog = Dialog(
        title=title,
        body=HSplit(
            [Label(text=text, dont_extend_height=True), radio_list],
            padding=1,
        ),
        buttons=[
            Button(text="确认", handler=ok_handler),
            Button(text="拒绝", handler=cancel_handler),
        ],
        with_background=True,
    )
    return _create_app(dialog, None).run()
