from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from .commands import PromptLike
from .repl_rendering import CLI_COLOR_ASSISTANT, CLI_COLOR_INFO, CLI_COLOR_USER
from xcode.agent.protocols import ContentBlock
from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ToolCallContent
from xcode.harness.session import SessionMetadataView, SessionRecord, SessionStore


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


def sync_agent_history(app: Any, store: SessionStore) -> None:
    """把当前 transcript 的文本历史同步到支持该接口的 agent。"""
    agent = getattr(app, "agent", None)
    load_history = getattr(agent, "load_history", None)
    if not callable(load_history):
        return
    records = store.load_records()
    load_history(records_to_agent_messages(records))
    _restore_contextual_state(app, records)
    set_notice = getattr(agent, "set_resumed_notice", None)
    if callable(set_notice):
        set_notice(
            "This conversation was resumed from a previous session. "
            "The transcript history above has been loaded as context. "
            "Continue the task as if the session was uninterrupted."
        )


def _restore_contextual_state(app: Any, records: list[SessionRecord]) -> None:
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
    """把 transcript 记录转换为模型可见的简化会话历史。

    事件级 assistant 记录（带工具调用）已经捕获了对应轮次的文本内容，
    因此跳过后续重复的"assistant"文本摘要记录，避免模型看到重复消息。
    """
    messages: list[AgentMessage] = []
    pending_tool_calls: list[ToolCallContent] = []
    seen_tool_call_ids: set[str] = set()
    has_event_assistant_since_last_user = False
    for record in records:
        if record.type == "user":
            messages.append(UserMessage(content=str(record.content)))
            has_event_assistant_since_last_user = False
            continue
        if record.type == "assistant":
            text = str(record.content).strip()
            if text and not has_event_assistant_since_last_user:
                messages.append(AssistantMessage(content=[TextContent(text=text)]))
            continue
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        event_type = str(record.content.get("type", ""))
        event_data = record.content.get("data")
        if event_type == "assistant":
            _append_tool_assistant_event(
                messages,
                event_data,
                pending_tool_calls,
                seen_tool_call_ids,
            )
            has_event_assistant_since_last_user = True
            continue
        if event_type == "tool_use":
            _queue_tool_use_event(event_data, pending_tool_calls, seen_tool_call_ids)
            continue
        if event_type == "tool_result":
            _append_tool_result_event(messages, event_data, pending_tool_calls)
    return messages


def _append_tool_assistant_event(
    messages: list[AgentMessage],
    event_data: object,
    pending_tool_calls: list[ToolCallContent],
    seen_tool_call_ids: set[str],
) -> None:
    """恢复包含工具调用的 assistant 事件。"""
    if not isinstance(event_data, list):
        return
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
        return
    messages.append(AssistantMessage(content=content))
    seen_tool_call_ids.update(tool_call_ids)
    pending_tool_calls[:] = [
        call for call in pending_tool_calls if call.id not in tool_call_ids
    ]


def _queue_tool_use_event(
    event_data: object,
    pending_tool_calls: list[ToolCallContent],
    seen_tool_call_ids: set[str],
) -> None:
    """在缺少 assistant 事件时缓存单独的 tool_use 事件。"""
    if not isinstance(event_data, dict):
        return
    tool_call = _tool_call_from_event_data(event_data)
    if tool_call is None or tool_call.id in seen_tool_call_ids:
        return
    pending_tool_calls.append(tool_call)
    seen_tool_call_ids.add(tool_call.id)


def _append_tool_result_event(
    messages: list[AgentMessage],
    event_data: object,
    pending_tool_calls: list[ToolCallContent],
) -> None:
    """恢复工具结果，并在需要时先补齐待恢复的工具调用。"""
    if pending_tool_calls:
        messages.append(AssistantMessage(content=list(pending_tool_calls)))
        pending_tool_calls.clear()
    if not isinstance(event_data, dict):
        return
    tool_use_id = str(event_data.get("tool_use_id", "")).strip()
    if not tool_use_id:
        return
    status = str(event_data.get("status", "ok"))
    messages.append(
        ToolResultMessage(
            tool_call_id=tool_use_id,
            content=str(event_data.get("content", "")),
            is_error=status not in {"ok", "interrupted"},
        )
    )


def _tool_call_from_block(block: dict[str, Any]) -> ToolCallContent | None:
    """从 assistant raw block 解析工具调用。"""
    if block.get("type") != "tool_use":
        return None
    tool_call_id = str(block.get("id", "")).strip()
    name = str(block.get("name", "")).strip()
    if not tool_call_id or not name:
        return None
    raw_input = block.get("input", {})
    arguments = raw_input if isinstance(raw_input, dict) else {}
    return ToolCallContent(id=tool_call_id, name=name, arguments=arguments)


def _tool_call_from_event_data(event_data: dict[str, Any]) -> ToolCallContent | None:
    """从 tool_use event data 解析工具调用。"""
    tool_call_id = str(event_data.get("id", "")).strip()
    name = str(event_data.get("name", "")).strip()
    if not tool_call_id or not name:
        return None
    raw_input = event_data.get("input", {})
    arguments = raw_input if isinstance(raw_input, dict) else {}
    return ToolCallContent(id=tool_call_id, name=name, arguments=arguments)


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
