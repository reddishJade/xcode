import json
from pathlib import Path

import pytest

from xcode.harness.session import SessionHistory, build_history_tools


def _claim(text: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "type": "inbox/claimed",
        "step": 0,
        "data": {
            "inbox_id": f"inbox-{len(text)}",
            "lane": "next_turn",
            "message": [
                {
                    "kind": "user",
                    "payload": {"content": text},
                }
            ],
            "source": "user",
            "display_text": text,
            "wake": True,
            "run_id": "run-1",
            "reason": "",
        },
        "correlation": {},
    }


def _write_session(sessions_dir: Path) -> None:
    sessions_dir.mkdir(parents=True)
    rows = [
        {
            "id": "u1",
            "parent_id": None,
            "type": "event",
            "content": _claim("The exact migration constraint is keep API v1."),
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "a1",
            "parent_id": "u1",
            "type": "assistant",
            "content": "I will preserve API v1.",
            "created_at": "2026-01-01T00:01:00+00:00",
        },
        {
            "id": "fork",
            "parent_id": "a1",
            "type": "assistant",
            "content": "Abandoned branch secret.",
            "created_at": "2026-01-01T00:02:00+00:00",
        },
        {
            "id": "u2",
            "parent_id": "a1",
            "type": "event",
            "content": _claim("Continue migration in src/api.py"),
            "created_at": "2026-01-01T00:03:00+00:00",
        },
        {
            "id": "w1",
            "parent_id": "u2",
            "type": "event",
            "content": {
                "schema_version": 2,
                "type": "context_window_reset",
                "step": 1,
                "data": {
                    "window_id": "window-2",
                    "trigger": "manual",
                    "replacement": [],
                },
                "correlation": {},
            },
            "created_at": "2026-01-01T00:04:00+00:00",
        },
    ]
    path = sessions_dir / "session-session-a.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (sessions_dir.parent / "session_index.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "id": "session-a",
                        "head_id": "w1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_history_search_only_reads_current_branch(tmp_path: Path) -> None:
    sessions_dir = tmp_path / ".xcode" / "sessions"
    _write_session(sessions_dir)
    history = SessionHistory(sessions_dir)
    history.set_session_id("session-a")

    hits = history.search("migration API")

    assert [hit.id for hit in hits] == ["u2", "u1", "a1"]
    assert history.search("Abandoned branch secret") == []


def test_history_around_returns_verbatim_neighbors(tmp_path: Path) -> None:
    sessions_dir = tmp_path / ".xcode" / "sessions"
    _write_session(sessions_dir)
    history = SessionHistory(sessions_dir)
    history.set_session_id("session-a")

    around = history.around("a1", before=1, after=1)

    assert [entry.id for entry in around] == ["u1", "a1", "u2"]
    assert around[0].text == "The exact migration constraint is keep API v1."
    assert "src/api.py" in around[2].text


def test_history_tool_exposes_window_search_read_and_around(tmp_path: Path) -> None:
    sessions_dir = tmp_path / ".xcode" / "sessions"
    _write_session(sessions_dir)
    history = SessionHistory(sessions_dir)
    history.set_session_id("session-a")
    (tool,) = build_history_tools(history)

    search = tool.handler({"operation": "search", "query": "constraint", "limit": 2})
    around = tool.handler(
        {"operation": "around", "message_id": "a1", "before": 0, "after": 0}
    )
    exact = tool.handler(
        {"operation": "read", "message_id": "a1", "offset": 2, "max_chars": 4}
    )
    windows = tool.handler({"operation": "list_windows"})

    assert "message_id=u1" in search
    assert "message_id=a1" in around
    assert "offset=2" in exact
    assert "next_offset=6" in exact
    assert exact.endswith("will")
    assert "window_id=window-2" in windows
    assert set(tool.schema["properties"]) == {
        "operation",
        "query",
        "message_id",
        "limit",
        "before",
        "after",
        "offset",
        "max_chars",
    }


def test_history_rejects_unsafe_session_id(tmp_path: Path) -> None:
    history = SessionHistory(tmp_path)

    with pytest.raises(ValueError, match="invalid session id"):
        history.set_session_id("../other")
