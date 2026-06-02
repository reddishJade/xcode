from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .commands import PromptLike
from .repl_rendering import CLI_COLOR_ASSISTANT, CLI_COLOR_INFO, CLI_COLOR_USER
from xcode.harness.session import SessionMetadataView, SessionStore


def resume_interactively(store: SessionStore, prompt_session: PromptLike) -> None:
    sessions = store.list_session_infos()
    if not sessions:
        print("No conversations found.")
        return
    print_sessions(sessions)
    choice = prompt_session.prompt("resume> ").strip()
    if not choice:
        print("Resume cancelled.")
        return
    selected = select_session(sessions, choice)
    if selected is None:
        print(f"No conversation matched: {choice}")
        return
    store.resume(selected.id)
    print(resumed_message(selected))
    print_loaded_history(store)


def resume_latest(store: SessionStore) -> SessionMetadataView | None:
    sessions = store.list_session_infos(limit=1)
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
        for record in store.load_records()
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
