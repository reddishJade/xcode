from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .markdown import MarkdownRenderer
from xcode.harness.session import SessionStore
from xcode.harness.observability import (
    PersistentPermissionStore,
    SessionPermissionPolicy,
)


PromptText = str | list[tuple[str, str]]


class PromptLike(Protocol):
    def prompt(self, prompt_text: PromptText) -> str: ...


@dataclass
class ReplState:
    mode: str = "act"
    verbose: bool = False
    approved_plan: str | None = None
    exit_pending: float = 0.0
    pending_partial: tuple[str, str] | None = None
    pending_inject: str | None = None


@dataclass
class CommandContext:
    store: SessionStore
    app: Any
    renderer: MarkdownRenderer
    state: ReplState
    prompt_session: PromptLike
    session_policy: SessionPermissionPolicy | None = None
    persistent_store: PersistentPermissionStore | None = None


CommandHandler = Callable[[str, CommandContext], bool]


@dataclass
class CommandEntry:
    handler: CommandHandler
    desc: str
    args_desc: str = ""
    accepts_args: bool = False
    visible: bool = True


def command_names(registry: dict[str, CommandEntry]) -> tuple[str, ...]:
    """从注册表派生可补全命令名。"""
    return tuple(name for name, entry in registry.items() if entry.visible)


def generate_help_text(registry: dict[str, CommandEntry]) -> str:
    """从注册表生成 HELP_TEXT。"""
    lines = ["Commands:"]
    for name, entry in registry.items():
        if not entry.visible:
            continue
        lines.append(f"  {name:<11} {entry.desc}")
        if entry.args_desc:
            lines.append(f"  {name} {entry.args_desc}")
    lines.extend(
        [
            "",
            "Press Shift+Enter for a newline. If your terminal does not send Shift+Enter,",
            "use Esc Enter as the fallback accepted by prompt_toolkit.",
            "Use Tab to complete slash commands, /tool names, and @file references.",
        ]
    )
    return "\n".join(lines)
