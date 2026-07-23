from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .commands import PromptLike
from .repl_rendering import CLI_COLOR_ASSISTANT, CLI_COLOR_INFO, CLI_COLOR_USER
from xcode.agent.types import ContentBlock
from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ToolArguments, ToolCallContent
from xcode.harness.session import (
    SessionEntry as SessionRecord,
    SessionInfoView as SessionMetadataView,
)
from xcode.harness.session import SessionStore
from xcode.harness.memory import load_session_checkpoint


@dataclass(frozen=True)
class _AssistantTranscriptEvent:
    content: tuple[ContentBlock, ...]
    tool_call_ids: frozenset[str]


@dataclass(frozen=True)
class _ToolUseTranscriptEvent:
    tool_call: ToolCallContent


@dataclass(frozen=True)
class _ToolResultTranscriptEvent:
    result: ToolResultMessage


_TranscriptEvent = (
    _AssistantTranscriptEvent | _ToolUseTranscriptEvent | _ToolResultTranscriptEvent
)


def resume_interactively(
    store: SessionStore, prompt_session: PromptLike, show_history: bool = True
) -> None:
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


def sync_agent_history(app: object, store: SessionStore) -> None:
    """把当前 transcript 的文本历史同步到支持该接口的 agent。"""
    sync_compaction_source(app, store, None)
    agent = getattr(app, "agent", None)
    load_history = getattr(agent, "load_history", None)
    if not callable(load_history):
        return
    records = store.build_branch()
    messages, rebuilt = _resume_messages(agent, store.session_id, records)
    load_history(messages)
    _restore_contextual_state(app, records)
    set_notice = getattr(agent, "set_resumed_notice", None)
    if callable(set_notice):
        if rebuilt:
            set_notice(
                "This long-running session was rebuilt from its latest checkpoint "
                "plus the verbatim transcript tail. Continue from the recorded next "
                "action without asking the user to restate the goal."
            )
        else:
            set_notice(
                "This conversation was resumed from a previous session. "
                "The transcript history above has been loaded as context. "
                "Continue the task as if the session was uninterrupted."
            )


def _resume_messages(
    agent: object,
    session_id: str,
    records: list[SessionRecord],
) -> tuple[list[AgentMessage], bool]:
    """优先用 checkpoint + 原文 tail 恢复，失配时回退完整历史。"""
    compactor = getattr(agent, "compactor", None)
    checkpoint_dir = getattr(compactor, "checkpoint_dir", None)
    if not isinstance(checkpoint_dir, Path):
        return records_to_agent_messages(records), False
    checkpoint = load_session_checkpoint(checkpoint_dir, session_id)
    if checkpoint is None:
        return records_to_agent_messages(records), False
    boundary_index = next(
        (
            index
            for index, record in enumerate(records)
            if record.id == checkpoint.boundary_message_id
        ),
        None,
    )
    if boundary_index is None:
        return records_to_agent_messages(records), False
    tail = records[boundary_index:]
    seed = UserMessage(content=checkpoint.render_rebuild_prompt())
    return [seed, *records_to_agent_messages(tail)], True


def sync_compaction_source(
    app: object,
    store: SessionStore,
    message_id: str | None,
) -> None:
    """向 agent 绑定真实会话 ID 和触发当前回合的原消息 ID。"""
    agent = getattr(app, "agent", None)
    if agent is None:
        return
    agent.session_id = store.session_id
    setter = getattr(agent, "set_compaction_source_message_id", None)
    if callable(setter):
        setter(message_id)


def _restore_contextual_state(app: object, records: list[SessionRecord]) -> None:
    """从 transcript 记录中恢复 ContextualRetrievalState 的活跃上下文。"""
    contextual_state = getattr(app, "contextual_state", None)
    if contextual_state is None:
        return
    for record in records:
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        event_type = str(record.content.get("type", ""))
        event_data = record.content.get("data")
        if event_type == "file_references" and isinstance(event_data, list):
            for ref in event_data:
                path = ref.get("path", "") if isinstance(ref, dict) else ""
                if path:
                    contextual_state.record_file(path)
        elif event_type == "tool_result" and isinstance(event_data, dict):
            tool_name = str(event_data.get("tool_use_id", "") or "")
            content = str(event_data.get("content", "") or "")
            if tool_name:
                contextual_state.record_tool_result(tool_name, content)


def records_to_agent_messages(records: list[SessionRecord]) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    pending_tool_calls: list[ToolCallContent] = []
    seen_tool_call_ids: set[str] = set()
    event_assistant_texts: list[str] = []
    for record in records:
        if record.type == "user":
            messages.append(UserMessage(content=str(record.content)))
            event_assistant_texts.clear()
            continue
        if record.type == "assistant":
            text = str(record.content).strip()
            event_text = "\n\n".join(event_assistant_texts).strip()
            remaining_text = text
            if event_text and text.startswith(event_text):
                remaining_text = text[len(event_text) :].strip()
            if remaining_text:
                messages.append(
                    AssistantMessage(content=[TextContent(text=remaining_text)])
                )
            event_assistant_texts.clear()
            continue
        if record.type != "event":
            continue
        event = _transcript_event_from_content(record.content)
        if isinstance(event, _AssistantTranscriptEvent):
            _append_tool_assistant_event(
                messages, event, pending_tool_calls, seen_tool_call_ids
            )
            event_assistant_texts.extend(
                block.text for block in event.content if isinstance(block, TextContent)
            )
        elif isinstance(event, _ToolUseTranscriptEvent):
            _queue_tool_use_event(event, pending_tool_calls, seen_tool_call_ids)
        elif isinstance(event, _ToolResultTranscriptEvent):
            _append_tool_result_event(messages, event, pending_tool_calls)
    return messages


def _transcript_event_from_content(content: object) -> _TranscriptEvent | None:
    if not isinstance(content, dict):
        return None
    event_type = str(content.get("type", ""))
    event_data = content.get("data")
    if event_type == "assistant":
        return _assistant_event_from_data(event_data)
    if event_type == "tool_use":
        return _tool_use_event_from_data(event_data)
    if event_type == "tool_result":
        return _tool_result_event_from_data(event_data)
    return None


def _assistant_event_from_data(event_data: object) -> _AssistantTranscriptEvent | None:
    if not isinstance(event_data, list):
        return None
    content: list[ContentBlock] = []
    tool_call_ids: set[str] = set()
    for block in event_data:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = str(block.get("text", "")).strip()
            if text:
                content.append(TextContent(text=text))
            continue
        tool_call = _tool_call_from_block(block)
        if tool_call is None:
            continue
        content.append(tool_call)
        tool_call_ids.add(tool_call.id)
    if not tool_call_ids:
        return None
    return _AssistantTranscriptEvent(
        content=tuple(content),
        tool_call_ids=frozenset(tool_call_ids),
    )


def _tool_use_event_from_data(event_data: object) -> _ToolUseTranscriptEvent | None:
    if not isinstance(event_data, dict):
        return None
    tool_call = _tool_call_from_event_data(event_data)
    if tool_call is None:
        return None
    return _ToolUseTranscriptEvent(tool_call)


def _tool_result_event_from_data(
    event_data: object,
) -> _ToolResultTranscriptEvent | None:
    if not isinstance(event_data, dict):
        return None
    tool_use_id = str(event_data.get("tool_use_id", "")).strip()
    if not tool_use_id:
        return None
    status = str(event_data.get("status", "ok"))
    return _ToolResultTranscriptEvent(
        ToolResultMessage(
            tool_call_id=tool_use_id,
            content=str(event_data.get("content", "")),
            is_error=status not in {"ok", "interrupted"},
        )
    )


def _append_tool_assistant_event(
    messages: list[AgentMessage],
    event: _AssistantTranscriptEvent,
    pending_tool_calls: list[ToolCallContent],
    seen_tool_call_ids: set[str],
) -> None:
    messages.append(AssistantMessage(content=list(event.content)))
    seen_tool_call_ids.update(event.tool_call_ids)
    pending_tool_calls[:] = [
        call for call in pending_tool_calls if call.id not in event.tool_call_ids
    ]


def _queue_tool_use_event(
    event: _ToolUseTranscriptEvent,
    pending_tool_calls: list[ToolCallContent],
    seen_tool_call_ids: set[str],
) -> None:
    tool_call = event.tool_call
    if tool_call.id in seen_tool_call_ids:
        return
    pending_tool_calls.append(tool_call)
    seen_tool_call_ids.add(tool_call.id)


def _append_tool_result_event(
    messages: list[AgentMessage],
    event: _ToolResultTranscriptEvent,
    pending_tool_calls: list[ToolCallContent],
) -> None:
    if pending_tool_calls:
        messages.append(AssistantMessage(content=list(pending_tool_calls)))
        pending_tool_calls.clear()
    messages.append(event.result)


def _tool_call_from_block(block: dict[str, object]) -> ToolCallContent | None:
    if block.get("type") != "tool_use":
        return None
    tool_call_id = str(block.get("id", "")).strip()
    name = str(block.get("name", "")).strip()
    if not tool_call_id or not name:
        return None
    return ToolCallContent(
        id=tool_call_id,
        name=name,
        arguments=_tool_arguments(block.get("input")),
    )


def _tool_call_from_event_data(event_data: dict[str, object]) -> ToolCallContent | None:
    tool_call_id = str(event_data.get("id", "")).strip()
    name = str(event_data.get("name", "")).strip()
    if not tool_call_id or not name:
        return None
    return ToolCallContent(
        id=tool_call_id,
        name=name,
        arguments=_tool_arguments(event_data.get("input")),
    )


def _tool_arguments(raw_input: object) -> ToolArguments:
    if not isinstance(raw_input, dict):
        return {}
    return {str(key): value for key, value in raw_input.items()}


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
    for _, item in enumerate(sessions, start=1):
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
