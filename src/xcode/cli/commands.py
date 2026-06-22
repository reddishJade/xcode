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
    FileGrantStore,
    InMemoryGrantStore,
    PermissionPolicy,
)
from xcode.harness.snapshot import SnapshotStore


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
    session_grant_store: InMemoryGrantStore | None = None
    permanent_grant_store: FileGrantStore | None = None
    static_policy: PermissionPolicy | None = None
    restricted_dirs: tuple[str, ...] = ()
    snapshot_store: SnapshotStore | None = None


CommandHandler = Callable[[str, CommandContext], bool]


COMMAND_GROUP_SESSION_LIFECYCLE = "Session Lifecycle"
COMMAND_GROUP_SESSION_BRANCH = "Session Branches"
COMMAND_GROUP_SESSION_ROLLBACK = "Session Rollback"
COMMAND_GROUP_MODE = "Mode Control"
COMMAND_GROUP_MODEL = "Model Configuration"
COMMAND_GROUP_INFO = "Info Tools"
COMMAND_GROUP_EXIT = "Exit"

COMMAND_GROUP_ORDER: dict[str, int] = {
    COMMAND_GROUP_SESSION_LIFECYCLE: 1,
    COMMAND_GROUP_SESSION_BRANCH: 2,
    COMMAND_GROUP_SESSION_ROLLBACK: 3,
    COMMAND_GROUP_MODE: 4,
    COMMAND_GROUP_MODEL: 5,
    COMMAND_GROUP_INFO: 6,
    COMMAND_GROUP_EXIT: 7,
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
    """从注册表按分组生成 HELP_TEXT。"""
    lines = ["Commands:"]
    groups: dict[str, list[tuple[str, CommandEntry]]] = {}
    for name, entry in registry.items():
        if not entry.visible:
            continue
        g = entry.group or ""
        groups.setdefault(g, []).append((name, entry))
    sorted_groups = sorted(groups, key=lambda g: COMMAND_GROUP_ORDER.get(g, 99))
    for group in sorted_groups:
        lines.append(f"\n  {group}:")
        for name, entry in groups[group]:
            lines.append(f"    {name:<11} {entry.desc}")
            if entry.args_desc:
                lines.append(f"    {name} {entry.args_desc}")
    lines.extend(
        [
            "",
            "",
            "Prefix a line with ! to run it through the registered bash tool.",
            "Press Shift+Enter for a newline. If your terminal does not send Shift+Enter,",
            "use Esc Enter as the fallback accepted by prompt_toolkit.",
            "Use Tab to complete slash commands, /tool names, and @file references.",
        ]
    )
    return "\n".join(lines)
