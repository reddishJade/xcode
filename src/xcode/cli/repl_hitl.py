"""HITL 授权处理器：基于结构化 GrantStore 的权限授权。

取代旧的 SessionPermissionPolicy + PersistentPermissionStore 模式。
授权结果写入新格式 GrantRecord，旧格式只作为一次性迁读取回退。
"""

from __future__ import annotations

import asyncio
from queue import Queue
from threading import Thread

from .repl_tools import brief_input
from xcode.harness.observability import (
    ActionExtractor,
    FileGrantStore,
    HITLResult,
    InMemoryGrantStore,
    create_grant_record,
)
from xcode.harness.observability.permission_model import Action
from xcode.harness.skills import ToolInput, ToolSpec


class ReplHITLHandler:
    """基于结构化 GrantStore 的 HITL 授权处理器。

    读取顺序：
      1. session_grant_store（InMemoryGrantStore）
      2. permanent_grant_store（FileGrantStore）
      3. 交互式用户提示

    写入只写入新格式 GrantStore。旧格式从不写入。
    """

    def __init__(
        self,
        session_grant_store: InMemoryGrantStore,
        permanent_grant_store: FileGrantStore,
        prompt: object | None = None,
    ) -> None:
        self._session_store = session_grant_store
        self._permanent_store = permanent_grant_store
        self._prompt = prompt

    def __call__(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        action = ActionExtractor().extract(tool.name, action_input)

        # 尝试从结构化 grant 存储查找
        for target in action.targets:
            grant = self._session_store.lookup(action, target)
            if grant is not None:
                return HITLResult(
                    grant.decision,
                    "session" if grant.scope == "session" else grant.scope,
                )
            grant = self._permanent_store.lookup(action, target)
            if grant is not None:
                return HITLResult(grant.decision, "permanent")

        # 没有结构化 grant → 交互式提示
        return self._interactive_prompt(action, tool, action_input)

    def _interactive_prompt(
        self,
        action: Action,
        tool: ToolSpec,
        action_input: ToolInput,
    ) -> HITLResult:
        choice = _ask_hitl_choice(tool, action_input)
        return self._apply_choice(choice, action, tool, action_input)

    def _apply_choice(
        self,
        choice: str | None,
        action: Action,
        tool: ToolSpec,
        action_input: ToolInput,
    ) -> HITLResult:
        if choice == "允许（仅本次）":
            return HITLResult("allow", "once")
        if choice == "此次对话中允许":
            self._write_grants(action, decision="allow", scope="session")
            return HITLResult("allow", "session")
        if choice == "始终允许":
            self._write_grants(action, decision="allow", scope="permanent")
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once")

    def _write_grants(
        self,
        action: Action,
        *,
        decision: str,
        scope: str,
    ) -> None:
        store: InMemoryGrantStore | FileGrantStore | None = None
        if scope == "session":
            store = self._session_store
        elif scope == "permanent":
            store = self._permanent_store

        if store is None:
            return

        for target in action.targets:
            grant = create_grant_record(
                action,
                target,
                decision=decision,  # type: ignore[arg-type]
                scope=scope,  # type: ignore[arg-type]
            )
            store.add(grant)


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
