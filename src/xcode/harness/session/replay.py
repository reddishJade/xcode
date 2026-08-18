"""从 append-only session transcript 重建 agent 运行状态。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import (
    ContentBlock,
    TextContent,
    ToolArguments,
    ToolCallContent,
)
from xcode.harness.memory import load_session_checkpoint

from .tree_store import TreeSessionRepo
from .types import SessionEntry


class ReplayAgent(Protocol):
    """会话投影器恢复运行状态所需的 agent 接口。"""

    @property
    def compactor(self) -> object | None: ...

    @property
    def session_id(self) -> str: ...

    @session_id.setter
    def session_id(self, value: str) -> None: ...

    def set_history_session_id(self, session_id: str) -> None: ...

    def set_compaction_source_message_id(self, message_id: str | None) -> None: ...

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def restore_run_state_metadata(self, payload: object) -> None: ...

    def restore_goal_state(self, payload: object) -> None: ...

    def set_resumed_notice(self, notice: str) -> None: ...


class ContextualReplayState(Protocol):
    """回放时恢复文件和工具上下文所需的状态接口。"""

    def record_file(self, path: str) -> None: ...

    def record_tool_result(self, tool: str, content: str) -> None: ...

    def clear(self) -> None: ...


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


type _TranscriptEvent = (
    _AssistantTranscriptEvent | _ToolUseTranscriptEvent | _ToolResultTranscriptEvent
)


def replay_session(
    agent: ReplayAgent,
    store: TreeSessionRepo,
    contextual_state: ContextualReplayState | None = None,
) -> None:
    """把当前 session branch 投影回 agent 的内存状态。"""
    agent.session_id = store.session_id
    agent.set_history_session_id(store.session_id)
    agent.set_compaction_source_message_id(None)
    records = store.build_branch()
    messages, rebuilt = resume_messages(agent, store.session_id, records)
    agent.load_history(messages)
    run_state = latest_run_state(records)
    if run_state is not None:
        agent.restore_run_state_metadata(run_state)
    goal_state = latest_goal_state(records)
    if goal_state is not None:
        agent.restore_goal_state(goal_state)
    if contextual_state is not None:
        contextual_state.clear()
        restore_contextual_state(contextual_state, records)
    if rebuilt:
        agent.set_resumed_notice(
            "This long-running session was rebuilt from its latest checkpoint "
            "plus the verbatim transcript tail. Continue from the recorded next "
            "action without asking the user to restate the goal."
        )
    else:
        agent.set_resumed_notice(
            "This conversation was resumed from a previous session. "
            "The transcript history above has been loaded as context. "
            "Continue the task as if the session was uninterrupted."
        )


def latest_run_state(records: list[SessionEntry]) -> object | None:
    """读取当前 branch 最近一次 final event 中的 run_state。"""
    for record in reversed(records):
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        if record.content.get("type") != "final":
            continue
        data = record.content.get("data")
        if not isinstance(data, dict):
            continue
        run_state = data.get("run_state")
        if isinstance(run_state, dict):
            return run_state
    return None


def latest_goal_state(records: list[SessionEntry]) -> object | None:
    """按时间倒序读取 Goal 命令或 final 中的最新状态。"""
    for record in reversed(records):
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        event_type = record.content.get("type")
        data = record.content.get("data")
        if event_type == "goal_state" and isinstance(data, dict):
            return data
        if event_type != "final" or not isinstance(data, dict):
            continue
        run_state = data.get("run_state")
        if isinstance(run_state, dict) and isinstance(run_state.get("goal"), dict):
            return run_state["goal"]
    return None


def resume_messages(
    agent: ReplayAgent,
    session_id: str,
    records: list[SessionEntry],
) -> tuple[list[AgentMessage], bool]:
    """优先用 checkpoint 加原文 tail 恢复，失配时使用完整历史。"""
    checkpoint_dir = getattr(agent.compactor, "checkpoint_dir", None)
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


def restore_contextual_state(
    contextual_state: ContextualReplayState,
    records: list[SessionEntry],
) -> None:
    """从 transcript 恢复上下文检索状态。"""
    for record in records:
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        event_type = str(record.content.get("type", ""))
        event_data = record.content.get("data")
        if event_type == "file_references" and isinstance(event_data, list):
            for ref in event_data:
                path = ref.get("path", "") if isinstance(ref, dict) else ""
                if isinstance(path, str) and path:
                    contextual_state.record_file(path)
        elif event_type == "tool_result" and isinstance(event_data, dict):
            tool_name = str(event_data.get("tool_use_id", "") or "")
            content = str(event_data.get("content", "") or "")
            if tool_name:
                contextual_state.record_tool_result(tool_name, content)


def records_to_agent_messages(records: list[SessionEntry]) -> list[AgentMessage]:
    """将 session 事实账本投影为 provider 使用的消息历史。"""
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
    return _ToolUseTranscriptEvent(tool_call) if tool_call is not None else None


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


def _tool_call_from_event_data(
    event_data: dict[str, object],
) -> ToolCallContent | None:
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
