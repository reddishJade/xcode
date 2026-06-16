"""HITL 授权处理器：交互式用户提示桥接。

不执行 grant 查找或写入。PermissionEngine 负责授权检查与持久化。
"""

from __future__ import annotations

import asyncio
from queue import Queue
from threading import Thread

from .repl_tools import brief_input
from xcode.harness.observability import HITLResult
from xcode.harness.skills import ToolInput, ToolSpec


class ReplHITLHandler:
    """HITL 授权处理器——仅交互式提示。

    PermissionEngine 负责 grant 查找与写入。
    本类只桥接用户的选择并返回 HITLResult。
    """

    def __init__(self, prompt: object | None = None) -> None:
        self._prompt = prompt

    def __call__(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        choice = _ask_hitl_choice(tool, action_input)
        return self._apply_choice(choice)

    def _apply_choice(self, choice: str | None) -> HITLResult:
        if choice == "允许（仅本次）":
            return HITLResult("allow", "once")
        if choice == "此次对话中允许":
            return HITLResult("allow", "session")
        if choice == "始终允许":
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
        f"需要授权：{tool.name}\n指令：{brief}",
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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
