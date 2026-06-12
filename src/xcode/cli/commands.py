from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .app_contract import ReplApp
from .markdown import MarkdownRenderer
from xcode.harness.config import ExecutionMode
from xcode.harness.session import SessionStore
from xcode.harness.observability import (
    PermissionPolicy,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)


VerbosityLevel = Literal["normal", "verbose", "debug"]
PromptText = str | list[tuple[str, str]]


class PromptLike(Protocol):
    def prompt(self, prompt_text: PromptText) -> str: ...


@dataclass
class ReplState:
    mode: ExecutionMode = "act"
    verbosity: VerbosityLevel = "normal"
    approved_plan: str | None = None
    exit_pending: float = 0.0
    pending_partial: tuple[str, str] | None = None
    pending_inject: str | None = None
    queue_mode: bool = False


@dataclass
class CommandContext:
    store: SessionStore
    app: ReplApp
    renderer: MarkdownRenderer
    state: ReplState
    prompt_session: PromptLike
    project_root: Path
    session_policy: SessionPermissionPolicy | None = None
    persistent_store: PersistentPermissionStore | None = None
    static_policy: PermissionPolicy | None = None
    restricted_dirs: tuple[str, ...] = ()
    allowlist_mode: bool = False


CommandHandler = Callable[[str, CommandContext], bool]


COMMAND_GROUP_SESSION = "会话管理"
COMMAND_GROUP_MODE = "模式控制"
COMMAND_GROUP_MODEL = "模型配置"
COMMAND_GROUP_INFO = "信息工具"
COMMAND_GROUP_EXIT = "退出"

COMMAND_GROUP_ORDER: dict[str, int] = {
    COMMAND_GROUP_SESSION: 1,
    COMMAND_GROUP_MODE: 2,
    COMMAND_GROUP_MODEL: 3,
    COMMAND_GROUP_INFO: 4,
    COMMAND_GROUP_EXIT: 5,
}


@dataclass
class CommandEntry:
    handler: CommandHandler
    desc: str
    args_desc: str = ""
    accepts_args: bool = False
    visible: bool = True
    group: str = ""


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
            "Prefix a line with ! to run it through the registered bash tool.",
            "Press Shift+Enter for a newline. If your terminal does not send Shift+Enter,",
            "use Esc Enter as the fallback accepted by prompt_toolkit.",
            "Use Tab to complete slash commands, /tool names, and @file references.",
        ]
    )
    return "\n".join(lines)
