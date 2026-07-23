from pathlib import Path

from xcode.agent.messages import UserMessage
from xcode.cli.repl_sessions import _resume_messages
from xcode.harness.agent_runtime.compaction import LayeredCompactor
from xcode.harness.memory import (
    load_session_checkpoint,
    write_session_checkpoint,
)
from xcode.harness.session.types import SessionEntry


class _Compactor:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir


class _Agent:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.compactor = _Compactor(checkpoint_dir)


def _entry(
    entry_id: str,
    entry_type: str,
    content: str,
    parent_id: str | None,
) -> SessionEntry:
    return SessionEntry(
        id=entry_id,
        parent_id=parent_id,
        type=entry_type,
        content=content,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _summary(cycle: int) -> str:
    return (
        "[Compressed]\n"
        "## Goal\nComplete the long-running migration without changing API v1.\n\n"
        "## Constraints & Preferences\nPreserve the user's exact API constraint.\n\n"
        "## Progress\n"
        f"### Done\n- [x] Completed context cycle {cycle}.\n\n"
        "### In Progress\n- [ ] Continue implementation.\n\n"
        "## Key Decisions\n- Keep API v1 stable.\n\n"
        "## Next Steps\n1. Run integration tests.\n\n"
        "## Critical Context\nThe transcript remains authoritative."
    )


def test_checkpoint_is_isolated_by_session_and_preserves_full_summary(
    tmp_path: Path,
) -> None:
    summary = (
        "[Compressed]\n## Goal\nFinish migration\n\n"
        "## Constraints & Preferences\nDo not change the API.\n\n"
        "## Next Steps\nRun the integration tests."
    )
    checkpoint = write_session_checkpoint(
        tmp_path,
        session_id="session-a",
        boundary_message_id="message-2",
        summary=summary,
        modified_files={"src/api.py"},
    )
    write_session_checkpoint(
        tmp_path,
        session_id="session-b",
        boundary_message_id="message-9",
        summary="[Compressed]\n## Goal\nOther task",
    )

    restored = load_session_checkpoint(tmp_path, "session-a")

    assert restored == checkpoint
    assert restored is not None
    assert "Constraints & Preferences" in restored.body
    assert "src/api.py" in restored.body
    assert checkpoint.path != (
        load_session_checkpoint(tmp_path, "session-b").path  # type: ignore[union-attr]
    )


def test_resume_uses_checkpoint_plus_verbatim_tail(tmp_path: Path) -> None:
    write_session_checkpoint(
        tmp_path,
        session_id="session-a",
        boundary_message_id="u2",
        summary="[Compressed]\n## Goal\nComplete the refactor\n\n"
        "## Next Steps\nEdit the parser.",
    )
    records = [
        _entry("u1", "user", "old request", None),
        _entry("a1", "assistant", "old answer", "u1"),
        _entry("u2", "user", "keep this wording exactly", "a1"),
        _entry("a2", "assistant", "current answer", "u2"),
    ]

    messages, rebuilt = _resume_messages(
        _Agent(tmp_path),
        "session-a",
        records,
    )

    assert rebuilt
    assert isinstance(messages[0], UserMessage)
    assert "Complete the refactor" in str(messages[0].content)
    rendered = [str(message.content) for message in messages]
    assert "old request" not in rendered
    assert "keep this wording exactly" in rendered
    assert any("current answer" in content for content in rendered)


def test_resume_falls_back_when_boundary_is_not_on_branch(tmp_path: Path) -> None:
    write_session_checkpoint(
        tmp_path,
        session_id="session-a",
        boundary_message_id="missing",
        summary="[Compressed]\n## Goal\nStale",
    )
    records = [_entry("u1", "user", "full history", None)]

    messages, rebuilt = _resume_messages(
        _Agent(tmp_path),
        "session-a",
        records,
    )

    assert not rebuilt
    assert [str(message.content) for message in messages] == ["full history"]


def test_compaction_writes_session_scoped_checkpoint(tmp_path: Path) -> None:
    compactor = LayeredCompactor(
        checkpoint_dir=tmp_path,
        max_recent_messages=1,
        keep_recent_tokens=20,
    )
    compactor.set_source_session_id("session-a")
    compactor.set_source_message_id("u2")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial goal"},
        {"role": "assistant", "content": "first result"},
        {"role": "user", "content": "continue the task"},
    ]

    compacted = compactor(messages)
    checkpoint = load_session_checkpoint(tmp_path, "session-a")

    assert len(compacted) < len(messages)
    assert checkpoint is not None
    assert checkpoint.boundary_message_id == "u2"
    assert checkpoint.path == tmp_path / "session-a" / "checkpoint.md"


def test_degenerate_update_cannot_replace_usable_checkpoint(tmp_path: Path) -> None:
    original = write_session_checkpoint(
        tmp_path,
        session_id="session-a",
        boundary_message_id="u1",
        summary=_summary(1),
    )

    result = write_session_checkpoint(
        tmp_path,
        session_id="session-a",
        boundary_message_id="u2",
        summary="too short",
    )

    assert result == original
    assert load_session_checkpoint(tmp_path, "session-a") == original


def test_two_compact_cycles_roll_previous_state_forward(tmp_path: Path) -> None:
    seen: list[list[dict[str, object]]] = []

    def summarize(messages: list[dict[str, object]]) -> str:
        seen.append(messages)
        return _summary(len(seen))

    compactor = LayeredCompactor(
        checkpoint_dir=tmp_path,
        max_recent_messages=1,
        keep_recent_tokens=20,
        summarize_fn=summarize,
    )
    compactor.set_source_session_id("session-a")
    compactor.set_source_message_id("u1")
    first = compactor(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "start migration"},
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "continue"},
        ]
    )

    compactor.set_source_message_id("u2")
    second = compactor(
        [
            *first,
            {"role": "assistant", "content": "more work"},
            {"role": "user", "content": "continue again"},
        ]
    )
    checkpoint = load_session_checkpoint(tmp_path, "session-a")

    assert len(seen) == 2
    assert any(
        "Completed context cycle 1" in str(message.get("content", ""))
        for message in seen[1]
    )
    assert checkpoint is not None
    assert checkpoint.boundary_message_id == "u2"
    assert "Completed context cycle 2" in checkpoint.body
    assert "Completed context cycle 2" in str(second[1]["content"])
