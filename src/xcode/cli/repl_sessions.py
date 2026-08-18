from __future__ import annotations

import sys

import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .commands import PromptLike
from .repl_rendering import CLI_COLOR_ASSISTANT, CLI_COLOR_INFO, CLI_COLOR_USER
from xcode.harness.session import (
    SessionInfoView as SessionMetadataView,
    SessionStore,
)


def resume_interactively(
    store: SessionStore, prompt_session: PromptLike, show_history: bool = True
) -> None:
    del prompt_session
    sessions = store.list_infos()
    if not sessions:
        print("No conversations found.")
        return
    selected = select_session_interactively(sessions, "Select session to resume:")
    if selected is None:
        print("Resume cancelled.")
        return
    store.resume(selected.id)
    if show_history:
        print(resumed_message(selected))
        print_loaded_history(store)


def resume_latest(store: SessionStore) -> SessionMetadataView | None:
    sessions = store.list_infos(limit=1)
    if not sessions:
        return None
    store.resume(sessions[0].id)
    return sessions[0]


def select_session(
    sessions: list[SessionMetadataView],
    choice: str,
) -> SessionMetadataView | None:
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
    for item in sessions:
        if choice in {item.id, item.title}:
            return item
    return None


def print_sessions(sessions: list[SessionMetadataView]) -> None:
    id_to_index = {session.id: str(index) for index, session in enumerate(sessions, 1)}
    for index, item in enumerate(sessions, start=1):
        suffix = ""
        if item.parent_id and item.parent_id in id_to_index:
            suffix = f" (forked from #{id_to_index[item.parent_id]})"
        print(f"{index}. {item.title}{suffix}")
        if item.summary:
            print(f"   {item.summary}")


def select_session_interactively(
    sessions: list[SessionMetadataView],
    title: str,
) -> SessionMetadataView | None:
    """显示支持方向键、鼠标和数字键选择的会话列表。"""
    choices = _session_choices(sessions)
    if not choices:
        return None
    return _run_session_picker(title, choices)


def _session_choices(
    sessions: list[SessionMetadataView],
) -> list[tuple[SessionMetadataView, str]]:
    """构建会话选择项。"""
    id_to_index = {session.id: str(index) for index, session in enumerate(sessions, 1)}
    choices: list[tuple[SessionMetadataView, str]] = []
    for item in sessions:
        title = item.title
        if item.parent_id and item.parent_id in id_to_index:
            title += f" (forked from #{id_to_index[item.parent_id]})"
        if item.summary:
            title += f" - {item.summary}"
        choices.append((item, title[:120]))
    return choices


def _run_session_picker(
    title: str,
    choices: list[tuple[SessionMetadataView, str]],
) -> SessionMetadataView | None:
    """显示会话选择器。"""
    questionary_choices = [
        questionary.Choice(title=label, value=session) for session, label in choices
    ]
    return questionary.select(title, choices=questionary_choices).ask()


def current_view(store: SessionStore) -> SessionMetadataView:
    metadata = store.current_metadata()
    if metadata is not None:
        return SessionMetadataView(
            id=metadata.id,
            title=metadata.title,
            summary=metadata.summary,
            updated_at=metadata.updated_at,
            path=store.current_path,
        )
    session_id = store.current_path.stem.removeprefix("session-")
    return SessionMetadataView(
        id=session_id,
        title=f"Session {session_id}",
        summary="No summary available.",
        updated_at="",
        path=store.current_path,
    )


def resumed_message(view: SessionMetadataView) -> str:
    return f"Resumed conversation: {view.title}"


def print_loaded_history(store: SessionStore) -> None:
    console = Console(file=sys.stdout)
    records = [
        record
        for record in store.build_branch()
        if record.type in {"user", "assistant"} and str(record.content).strip()
    ]
    if not records:
        return
    console.print(
        Text(
            f"  • loaded {len(records)} message(s) from this session branch",
            style=CLI_COLOR_INFO,
        )
    )
    for record in records:
        if record.type == "assistant":
            console.print(Text("assistant:", style=CLI_COLOR_ASSISTANT))
            console.print(Markdown(str(record.content)))
        else:
            console.print(Text(f"user: {record.content}", style=CLI_COLOR_USER))


def print_saved_conversation(store: SessionStore) -> None:
    metadata = store.update_summary()
    if metadata is not None:
        print(f"Conversation saved: {metadata.title}")
