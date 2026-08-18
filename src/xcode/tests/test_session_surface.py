"""Session current surface 投影测试。"""

from __future__ import annotations

from typing import Any, cast

import pytest

from xcode.agent.messages import (
    AgentMessage,
    AssistantMessage,
    CompactionSummaryMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ToolCallContent
from xcode.harness.session.replay import replay_session
from xcode.harness.session.surface import (
    InvalidSessionSurfaceError,
    decode_surface_messages,
    encode_surface_messages,
    project_session_surface,
    surface_digest,
)
from xcode.harness.session.types import SessionEntry


def _entry(
    entry_id: str,
    entry_type: str,
    content: Any,
    parent_id: str | None,
) -> SessionEntry:
    return SessionEntry(
        id=entry_id,
        parent_id=parent_id,
        type=entry_type,
        content=content,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _replacement_event(
    entry_id: str,
    parent_id: str,
    messages: list[AgentMessage],
    generation: int,
    source_entry_ids: list[str],
) -> SessionEntry:
    return _entry(
        entry_id,
        "event",
        {
            "schema_version": 2,
            "type": "compaction",
            "step": 2,
            "data": {
                "generation": generation,
                "source_entry_ids": source_entry_ids,
                "surface_sha256": surface_digest(messages),
                "replacement": encode_surface_messages(messages),
            },
            "correlation": {},
        },
        parent_id,
    )


def test_surface_message_codec_round_trips_tool_pairs() -> None:
    messages: list[AgentMessage] = [
        UserMessage(content="inspect"),
        AssistantMessage(
            content=[
                ToolCallContent(
                    id="call-1",
                    name="read_file",
                    arguments={"path": "README.md"},
                )
            ]
        ),
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read_file",
            content="contents",
        ),
    ]

    restored = decode_surface_messages(encode_surface_messages(messages))

    assert restored == messages


def test_projection_applies_latest_replacement_then_verbatim_tail() -> None:
    replacement: list[AgentMessage] = [
        CompactionSummaryMessage(summary="completed setup"),
        UserMessage(content="keep this exact constraint"),
    ]
    records = [
        _entry("u1", "user", "old request", None),
        _entry("a1", "assistant", "old answer", "u1"),
        _replacement_event("c1", "a1", replacement, 1, ["u1", "a1"]),
        _entry("u2", "user", "next action", "c1"),
        _entry("a2", "assistant", "done", "u2"),
    ]

    surface = project_session_surface(records)

    assert surface.generation == 1
    assert surface.replacement_entry_id == "c1"
    assert [type(message) for message in surface.messages] == [
        CompactionSummaryMessage,
        UserMessage,
        UserMessage,
        AssistantMessage,
    ]
    assert "old request" not in str(surface.messages)
    assert surface.messages[-1].content == [TextContent(text="done")]


def test_projection_rejects_untyped_or_unbalanced_replacement() -> None:
    with pytest.raises(InvalidSessionSurfaceError, match="typed surface"):
        project_session_surface(
            [
                _entry(
                    "c1",
                    "event",
                    {"type": "compaction", "data": {"summary": "legacy"}},
                    None,
                )
            ]
        )

    unbalanced: list[AgentMessage] = [
        AssistantMessage(
            content=[ToolCallContent(id="call-1", name="bash", arguments={})]
        )
    ]
    with pytest.raises(InvalidSessionSurfaceError, match="unresolved tool calls"):
        encode_surface_messages(unbalanced)


class _ReplayAgent:
    def __init__(self) -> None:
        self.session_id = "local"
        self.loaded: list[AgentMessage] = []
        self.notice = ""

    def set_history_session_id(self, _session_id: str) -> None:
        pass

    def load_history(self, messages: list[AgentMessage]) -> None:
        self.loaded = messages

    def restore_run_state_metadata(self, _payload: object) -> None:
        pass

    def restore_goal_state(self, _payload: object) -> None:
        pass

    def set_resumed_notice(self, notice: str) -> None:
        self.notice = notice


class _Store:
    session_id = "session-a"

    def __init__(self, records: list[SessionEntry]) -> None:
        self._records = records

    def build_branch(self) -> list[SessionEntry]:
        return self._records


def test_replay_uses_durable_surface_without_external_checkpoint() -> None:
    replacement: list[AgentMessage] = [
        CompactionSummaryMessage(summary="durable current state"),
        UserMessage(content="continue migration"),
    ]
    records = [
        _entry("u1", "user", "old", None),
        _replacement_event("c1", "u1", replacement, 1, ["u1"]),
    ]
    agent = _ReplayAgent()

    replay_session(agent, cast(Any, _Store(records)))

    assert agent.loaded == replacement
    assert "surface replacement" in agent.notice
