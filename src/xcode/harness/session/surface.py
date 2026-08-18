"""从 session 事实事件投影模型可见的当前 surface。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import ToolCallContent

from .types import JsonValue, SessionEntry


class InvalidSessionSurfaceError(ValueError):
    """持久化 surface replacement 无法形成合法模型历史。"""


@dataclass(frozen=True)
class SessionSurface:
    """由事实日志计算出的当前模型历史。"""

    messages: tuple[AgentMessage, ...]
    generation: int = 0
    replacement_entry_id: str | None = None


_MESSAGE_TYPES = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AssistantMessage,
    "tool_result": ToolResultMessage,
    "compaction_summary": CompactionSummaryMessage,
    "branch_summary": BranchSummaryMessage,
}


def encode_surface_messages(messages: list[AgentMessage]) -> list[JsonValue]:
    """以带显式类型标签的格式编码内部消息。"""
    validate_tool_pairing(messages)
    encoded: list[JsonValue] = []
    for message in messages:
        payload = message.model_dump(mode="json")
        encoded.append(
            {
                "kind": _message_kind(message),
                "payload": cast(JsonValue, payload),
            }
        )
    return encoded


def decode_surface_messages(value: object) -> list[AgentMessage]:
    """严格解码 surface replacement，不接受未标记旧格式。"""
    if not isinstance(value, list):
        raise InvalidSessionSurfaceError("surface replacement must be a list")
    messages: list[AgentMessage] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise InvalidSessionSurfaceError(
                f"surface message {index} must be an object"
            )
        kind = item.get("kind")
        payload = item.get("payload")
        model = _MESSAGE_TYPES.get(str(kind))
        if model is None or not isinstance(payload, dict):
            raise InvalidSessionSurfaceError(
                f"surface message {index} has invalid kind or payload"
            )
        try:
            messages.append(model.model_validate(payload))
        except ValueError as exc:
            raise InvalidSessionSurfaceError(
                f"surface message {index} is invalid"
            ) from exc
    validate_tool_pairing(messages)
    return messages


def surface_digest(messages: list[AgentMessage]) -> str:
    """计算稳定 surface 指纹。"""
    payload = json.dumps(
        encode_surface_messages(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def project_session_surface(records: list[SessionEntry]) -> SessionSurface:
    """依次应用 transcript facts 与 surface replacement。"""
    messages: list[AgentMessage] = []
    pending_tool_calls: list[ToolCallContent] = []
    seen_tool_call_ids: set[str] = set()
    event_assistant_texts: list[str] = []
    generation = 0
    replacement_entry_id: str | None = None

    for record_index, record in enumerate(records):
        if record.type == "user":
            messages.append(UserMessage(content=str(record.content)))
            event_assistant_texts.clear()
            continue
        if record.type == "assistant":
            _append_plain_assistant(messages, record, event_assistant_texts)
            event_assistant_texts.clear()
            continue
        if record.type != "event" or not isinstance(record.content, dict):
            continue

        event_type = record.content.get("type")
        data = record.content.get("data")
        if event_type == "compaction":
            if not isinstance(data, dict) or "replacement" not in data:
                raise InvalidSessionSurfaceError(
                    "compaction event is missing a typed surface replacement"
                )
            source_entry_ids = data.get("source_entry_ids")
            expected_source_ids = [item.id for item in records[:record_index]]
            if source_entry_ids != expected_source_ids:
                raise InvalidSessionSurfaceError(
                    "compaction source entry IDs do not match the branch prefix"
                )
            messages = decode_surface_messages(data["replacement"])
            expected_digest = data.get("surface_sha256")
            if expected_digest != surface_digest(messages):
                raise InvalidSessionSurfaceError(
                    "compaction surface fingerprint does not match replacement"
                )
            raw_generation = data.get("generation")
            if not isinstance(raw_generation, int) or raw_generation <= generation:
                raise InvalidSessionSurfaceError(
                    "compaction surface generation must increase"
                )
            generation = raw_generation
            replacement_entry_id = record.id
            pending_tool_calls.clear()
            seen_tool_call_ids = _tool_call_ids(messages)
            event_assistant_texts.clear()
            continue

        event = _transcript_event(data, str(event_type))
        if isinstance(event, _AssistantEvent):
            _append_tool_assistant(
                messages,
                event,
                pending_tool_calls,
                seen_tool_call_ids,
            )
            event_assistant_texts.extend(event.texts)
        elif isinstance(event, ToolCallContent):
            if event.id not in seen_tool_call_ids:
                pending_tool_calls.append(event)
                seen_tool_call_ids.add(event.id)
        elif isinstance(event, ToolResultMessage):
            _append_tool_result(messages, event, pending_tool_calls)

    return SessionSurface(
        messages=tuple(messages),
        generation=generation,
        replacement_entry_id=replacement_entry_id,
    )


def validate_tool_pairing(messages: list[AgentMessage]) -> None:
    """确保 replacement 不会切开 assistant tool call/result 对。"""
    pending: set[str] = set()
    completed: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if not isinstance(block, ToolCallContent):
                    continue
                if block.id in pending or block.id in completed:
                    raise InvalidSessionSurfaceError(
                        f"duplicate tool call in surface: {block.id}"
                    )
                pending.add(block.id)
        elif isinstance(message, ToolResultMessage):
            if message.tool_call_id not in pending:
                raise InvalidSessionSurfaceError(
                    f"orphan tool result in surface: {message.tool_call_id}"
                )
            pending.remove(message.tool_call_id)
            completed.add(message.tool_call_id)
    if pending:
        unresolved = ", ".join(sorted(pending))
        raise InvalidSessionSurfaceError(
            f"surface ends with unresolved tool calls: {unresolved}"
        )


@dataclass(frozen=True)
class _AssistantEvent:
    content: tuple[object, ...]
    tool_call_ids: frozenset[str]
    texts: tuple[str, ...]


def _message_kind(message: AgentMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, UserMessage):
        return "user"
    if isinstance(message, AssistantMessage):
        return "assistant"
    if isinstance(message, ToolResultMessage):
        return "tool_result"
    if isinstance(message, CompactionSummaryMessage):
        return "compaction_summary"
    return "branch_summary"


def _append_plain_assistant(
    messages: list[AgentMessage],
    record: SessionEntry,
    event_texts: list[str],
) -> None:
    from xcode.agent.types import TextContent

    text = str(record.content).strip()
    event_text = "\n\n".join(event_texts).strip()
    remaining = (
        text[len(event_text) :].strip()
        if event_text and text.startswith(event_text)
        else text
    )
    if remaining:
        messages.append(AssistantMessage(content=[TextContent(text=remaining)]))


def _transcript_event(data: object, event_type: str) -> object | None:
    if event_type == "assistant":
        return _assistant_event(data)
    if event_type == "tool_use":
        return _tool_call(data)
    if event_type == "tool_result":
        return _tool_result(data)
    return None


def _assistant_event(data: object) -> _AssistantEvent | None:
    from xcode.agent.types import ContentBlock, TextContent

    if not isinstance(data, list):
        return None
    content: list[ContentBlock] = []
    ids: set[str] = set()
    texts: list[str] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "text":
            text = str(raw.get("text", "")).strip()
            if text:
                content.append(TextContent(text=text))
                texts.append(text)
            continue
        call = _tool_call(raw)
        if call is not None:
            content.append(call)
            ids.add(call.id)
    if not ids:
        return None
    return _AssistantEvent(tuple(content), frozenset(ids), tuple(texts))


def _tool_call(data: object) -> ToolCallContent | None:
    if not isinstance(data, dict):
        return None
    tool_call_id = str(data.get("id", "")).strip()
    name = str(data.get("name", "")).strip()
    arguments = data.get("input", {})
    if not tool_call_id or not name or not isinstance(arguments, dict):
        return None
    return ToolCallContent(id=tool_call_id, name=name, arguments=arguments)


def _tool_result(data: object) -> ToolResultMessage | None:
    from xcode.agent.types import TextContent, parse_tool_render_intent

    if not isinstance(data, dict):
        return None
    tool_call_id = str(data.get("tool_use_id", "")).strip()
    if not tool_call_id:
        return None
    status = str(data.get("status", "ok"))
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        content=[TextContent(text=str(data.get("content", "")))],
        is_error=status != "ok",
        render_intent=parse_tool_render_intent(data.get("render_intent")),
    )


def _append_tool_assistant(
    messages: list[AgentMessage],
    event: _AssistantEvent,
    pending: list[ToolCallContent],
    seen: set[str],
) -> None:
    from xcode.agent.types import ContentBlock

    inline_calls = [
        block
        for block in event.content
        if isinstance(block, ToolCallContent) and block.id not in seen
    ]
    calls = [*pending, *inline_calls]
    if not calls:
        return
    content = [
        block for block in event.content if not isinstance(block, ToolCallContent)
    ]
    content.extend(calls)
    messages.append(AssistantMessage(content=cast(list[ContentBlock], content)))
    seen.update(call.id for call in calls)
    pending.clear()


def _append_tool_result(
    messages: list[AgentMessage],
    result: ToolResultMessage,
    pending: list[ToolCallContent],
) -> None:
    if pending:
        messages.append(AssistantMessage(content=list(pending)))
        pending.clear()
    messages.append(result)


def _tool_call_ids(messages: list[AgentMessage]) -> set[str]:
    return {
        block.id
        for message in messages
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, ToolCallContent)
    }
