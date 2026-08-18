"""当前 session 原始轨迹的只读检索。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re

from xcode.agent.types import ToolInput, ToolSpec

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class HistoryEntry:
    """用于召回的一条原始 session 记录。"""

    id: str
    parent_id: str | None
    type: str
    content: object
    created_at: str

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, ensure_ascii=False, sort_keys=True)


class SessionHistory:
    """读取当前 branch 的 lossless JSONL 历史。"""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.session_id: str | None = None

    def set_session_id(self, session_id: str) -> None:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError(f"invalid session id: {session_id!r}")
        self.session_id = session_id

    def search(self, query: str, *, limit: int = 5) -> list[HistoryEntry]:
        """按关键词相关性搜索当前 branch。"""
        terms = _tokens(query)
        if not terms:
            return []
        phrase = query.strip().casefold()
        scored: list[tuple[int, int, HistoryEntry]] = []
        for index, entry in enumerate(self._branch()):
            text = entry.text.casefold()
            matches = sum(text.count(term) for term in terms)
            if matches == 0:
                continue
            phrase_bonus = 10 if phrase and phrase in text else 0
            scored.append((matches + phrase_bonus, index, entry))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [entry for _, _, entry in scored[: min(max(limit, 1), 20)]]

    def around(
        self,
        message_id: str,
        *,
        before: int = 3,
        after: int = 3,
    ) -> list[HistoryEntry]:
        """返回指定记录附近的 branch 原文。"""
        branch = self._branch()
        index = next(
            (
                position
                for position, entry in enumerate(branch)
                if entry.id == message_id
            ),
            None,
        )
        if index is None:
            return []
        start = max(0, index - min(max(before, 0), 20))
        end = min(len(branch), index + min(max(after, 0), 20) + 1)
        return branch[start:end]

    def _branch(self) -> list[HistoryEntry]:
        session_id = self.session_id
        if session_id is None:
            return []
        path = self.sessions_dir / f"session-{session_id}.jsonl"
        entries = _read_entries(path)
        if not entries:
            return []
        head_id = self._head_id(session_id)
        by_id = {entry.id: entry for entry in entries}
        if head_id is None or head_id not in by_id:
            return entries
        branch: list[HistoryEntry] = []
        current: str | None = head_id
        seen: set[str] = set()
        while current and current in by_id and current not in seen:
            seen.add(current)
            entry = by_id[current]
            branch.append(entry)
            current = entry.parent_id
        branch.reverse()
        return branch

    def _head_id(self, session_id: str) -> str | None:
        index_dir = (
            self.sessions_dir.parent
            if self.sessions_dir.name == "sessions"
            else self.sessions_dir
        )
        path = index_dir / "session_index.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        for item in sessions:
            if isinstance(item, dict) and item.get("id") == session_id:
                value = item.get("head_id")
                return str(value) if value else None
        return None


def build_history_tools(history: SessionHistory) -> tuple[ToolSpec, ...]:
    """构建当前 session 的 search/around 工具。"""

    def handle(
        data: ToolInput,
        _on_update: Callable[[str], None] | None = None,
    ) -> str:
        operation = str(data.get("operation", "search"))
        if operation == "search":
            query = str(data.get("query", "")).strip()
            if not query:
                return "query is required for history search"
            entries = history.search(query, limit=_bounded(data.get("limit"), 5))
        elif operation == "around":
            message_id = str(data.get("message_id", "")).strip()
            if not message_id:
                return "message_id is required for history around"
            entries = history.around(
                message_id,
                before=_bounded(data.get("before"), 3),
                after=_bounded(data.get("after"), 3),
            )
        else:
            return "operation must be one of: search, around"
        if not entries:
            return "No matching history in the current session."
        return "\n\n".join(_render_entry(entry) for entry in entries)

    return (
        ToolSpec(
            name="history",
            description=(
                "Search the current session's lossless transcript or read records "
                "around a message ID after context rebuild."
            ),
            input_hint=(
                'JSON: {"operation":"search","query":"timeout","limit":5} or '
                '{"operation":"around","message_id":"abc123","before":3,"after":3}'
            ),
            handler=handle,
            schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["search", "around"]},
                    "query": {"type": "string"},
                    "message_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "before": {"type": "integer", "minimum": 0, "maximum": 20},
                    "after": {"type": "integer", "minimum": 0, "maximum": 20},
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            prompt_snippet=(
                "Use history search/around when a detail predates the current "
                "durable surface replacement or compacted context."
            ),
        ),
    )


def _read_entries(path: Path) -> list[HistoryEntry]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: list[HistoryEntry] = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("id", "")).strip()
        entry_type = str(item.get("type", "")).strip()
        if not entry_id or not entry_type:
            continue
        parent = item.get("parent_id")
        entries.append(
            HistoryEntry(
                id=entry_id,
                parent_id=str(parent) if parent else None,
                type=entry_type,
                content=item.get("content"),
                created_at=str(item.get("created_at", "")),
            )
        )
    return entries


def _render_entry(entry: HistoryEntry) -> str:
    content = entry.text
    if len(content) > 2000:
        content = (
            content[:1200]
            + f"\n[…{len(content) - 2000} chars omitted…]\n"
            + content[-800:]
        )
    return f"[message_id={entry.id} type={entry.type} at={entry.created_at}]\n{content}"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(re.findall(r"[a-z0-9_./:-]+|[\u3400-\u9fff]", text.casefold()))
    )


def _bounded(value: object, default: int) -> int:
    try:
        return min(max(int(value), 0), 20)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
