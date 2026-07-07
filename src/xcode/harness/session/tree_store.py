"""树结构会话存储：每条 entry 带 id/parent_id，支持会话内分支。

文件格式：每个会话一个 .jsonl 文件，每行是一个树 entry：
  {"id":"e1","parent_id":null,"type":"user","content":"...","created_at":"..."}
  {"id":"e2","parent_id":"e1","type":"assistant","content":"...","created_at":"..."}

head_id 记录在 session_index.json 的 metadata 中。
分支只需在同文件中追加不同 parent_id 的 entry。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

import filelock
from pydantic import BaseModel, ConfigDict, ValidationError

from ..skill_activation import is_skill_activation_content
from .types import (
    JsonValue,
    SessionEntry,
    SessionInfoView,
    TreeNode,
)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SUMMARY_USER_CHARS = 120
SUMMARY_ASSISTANT_CHARS = 180
SUMMARY_TITLE_CHARS = 160


class TreeEntryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    parent_id: str | None = None
    type: str
    content: JsonValue
    created_at: str


class TreeMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    title: str
    summary: str
    project_path: str
    transcript_path: str
    created_at: str
    updated_at: str
    parent_id: str | None = None
    head_id: str | None = None  # 当前活动叶节点


class TreeSessionRepo:
    """树结构会话存储。

    API 兼容 SessionStore（xcode.harness.session.SessionStore）。
    """

    def __init__(
        self,
        sessions_dir: Path,
        project_root: Path | None = None,
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = (project_root or sessions_dir).resolve()
        index_dir = (
            self.sessions_dir.parent
            if self.sessions_dir.name == "sessions"
            else self.sessions_dir
        )
        self.index_path = index_dir / "session_index.json"
        self._lock = filelock.FileLock(
            str(index_dir / "session_store.lock"),
            timeout=lock_timeout_seconds,
        )
        self.current_path = self._new_path()
        self.artifacts_dir = self.project_root / ".local" / "session_artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 公共 API ──

    @property
    def session_id(self) -> str:
        return self._session_id(self.current_path)

    def append(self, record_type: str, content: JsonValue) -> str:
        """追加一条树 entry，自动设置 parent_id 为当前 head。"""
        with self._lock:
            head_id = self._load_head_id()
            entry_id = uuid4().hex[:12]
            entry = TreeEntryModel(
                id=entry_id,
                parent_id=head_id,
                type=record_type,
                content=content,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            with self.current_path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
            self._save_head_id(entry_id)
            if record_type == "user":
                self.ensure_metadata(str(content))
            return entry_id

    def read_entries(self) -> list[SessionEntry]:
        """读取当前会话的全部 entry。"""
        if not self.current_path.exists():
            return []
        entries: list[SessionEntry] = []
        for line in self.current_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                model = TreeEntryModel.model_validate(data)
                entries.append(
                    SessionEntry(
                        id=model.id,
                        parent_id=model.parent_id,
                        type=model.type,
                        content=model.content,
                        created_at=model.created_at,
                    )
                )
            except (ValidationError, json.JSONDecodeError, TypeError):
                continue
        return entries

    def build_branch(self) -> list[SessionEntry]:
        """从当前 head 回溯到根，返回从根到叶的路径。"""
        entries = self.read_entries()
        by_id = {e.id: e for e in entries}
        head_id = self._load_head_id()
        if head_id is None or head_id not in by_id:
            return entries[-200:] if entries else []
        chain: list[SessionEntry] = []
        current: str | None = head_id
        while current and current in by_id:
            chain.append(by_id[current])
            parent = by_id[current].parent_id
            if parent == current:
                break
            current = parent
        chain.reverse()
        return chain

    def fork_into(self, title: str = "", summary: str = "") -> TreeSessionRepo:
        """从当前 head 分叉，创建新会话（新文件，拷贝 entry）。"""
        parent = self.ensure_metadata()
        fork_path = self._new_path()
        if self.current_path.exists():
            shutil.copy2(self.current_path, fork_path)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        meta = TreeMetadata(
            id=self._session_id(fork_path),
            title=title or f"Fork of {parent.title}",
            summary=summary or parent.summary,
            project_path=parent.project_path,
            transcript_path=str(fork_path),
            created_at=now,
            updated_at=now,
            parent_id=parent.id,
        )
        self._upsert_metadata(meta)
        fork = TreeSessionRepo.__new__(TreeSessionRepo)
        fork.sessions_dir = self.sessions_dir
        fork.project_root = self.project_root
        fork.index_path = self.index_path
        fork._lock = self._lock
        fork.current_path = fork_path
        fork.artifacts_dir = self.artifacts_dir
        return fork

    def clear(self) -> None:
        with self._lock:
            self.current_path = self._new_path()

    def resume(self, target: Path | str) -> None:
        path = self._resolve_target(target)
        if not path.exists():
            raise ValueError(f"session does not exist: {path}")
        self.current_path = path

    def switch_branch(self, target: str) -> SessionInfoView:
        """切换分支并设置 head_id。"""
        self.resume(target)
        meta = self.current_metadata()
        view = self._view_for_path(self.current_path, meta)
        return view

    def resume_latest(self) -> Path | None:
        sessions = [item.path for item in self.list_infos(limit=1)]
        if not sessions:
            return None
        self.current_path = sessions[0]
        return self.current_path

    def rewind_turns(self, turns: int = 1) -> int:
        """回退指定轮次：将 head_id 往回移动。"""
        with self._lock:
            branch = self.build_branch()
            user_indices = [
                i for i, e in enumerate(branch) if e.type == "user"
            ]
            if not user_indices:
                return 0
            target_idx = max(0, len(user_indices) - turns)
            target_entry = branch[user_indices[target_idx]]
            self._save_head_id(target_entry.id)
            return len(user_indices) - target_idx

    def user_turn_count(self) -> int:
        return sum(1 for e in self.read_entries() if e.type == "user")

    def compact_current_session(self, max_tool_result_chars: int = 200) -> int:
        """压缩当前会话，截断过长工具结果。"""
        with self._lock:
            entries = self.read_entries()
            if not entries:
                return 0
            compacted = 0
            new_data: list[dict] = []
            for e in entries:
                row = {
                    "id": e.id,
                    "parent_id": e.parent_id,
                    "type": e.type,
                    "content": e.content,
                    "created_at": e.created_at,
                }
                if (
                    e.type == "event"
                    and isinstance(e.content, dict)
                    and e.content.get("type") == "tool_result"
                ):
                    data = e.content.get("data")
                    if isinstance(data, dict) and "content" in data:
                        content_str = str(data["content"])
                        if is_skill_activation_content(content_str):
                            new_data.append(row)
                            continue
                        if len(content_str) > max_tool_result_chars:
                            data["content"] = (
                                "[Previous tool_result compacted; "
                                f"{len(content_str)} chars removed]"
                            )
                            compacted += 1
                new_data.append(row)
            if compacted > 0:
                with self.current_path.open("w", encoding="utf-8") as f:
                    for row in new_data:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return compacted

    def list_sessions(self, limit: int = 10) -> list[Path]:
        return [item.path for item in self.list_infos(limit=limit)]

    def list_infos(self, limit: int | None = 10) -> list[SessionInfoView]:
        known = {item.id: item for item in self._load_metadata()}
        views: list[SessionInfoView] = []
        for path in self._session_paths():
            meta = known.get(self._session_id(path))
            views.append(self._view_for_path(path, meta))
        sorted_views = sorted(views, key=lambda v: v.updated_at, reverse=True)
        if limit is not None:
            return sorted_views[:limit]
        return sorted_views

    def ensure_metadata(self, first_user_text: str | None = None) -> TreeMetadata:
        existing = self._metadata_for_path(self.current_path)
        if existing is not None:
            return existing
        now = datetime.now(UTC).isoformat(timespec="seconds")
        title = _make_title(first_user_text) if first_user_text else "Untitled conversation"
        meta = TreeMetadata(
            id=self._session_id(self.current_path),
            title=title,
            summary=_make_initial_summary(first_user_text) if first_user_text else "Conversation started.",
            project_path=str(self.project_root),
            transcript_path=str(self.current_path),
            created_at=now,
            updated_at=now,
        )
        self._upsert_metadata(meta)
        return meta

    def update_summary(self) -> TreeMetadata | None:
        with self._lock:
            meta = self._metadata_for_path(self.current_path)
            if meta is None:
                return None
            entries = self.build_branch()
            user_text = ""
            assistant_text: str | None = None
            for e in entries:
                if e.type == "user" and not user_text:
                    user_text = _collapse_text(str(e.content))
                elif e.type == "assistant" and assistant_text is None:
                    assistant_text = _collapse_text(str(e.content))
            summary = (
                _make_conversation_summary(user_text, assistant_text)
                if user_text
                else meta.summary
            )
            now = datetime.now(UTC).isoformat(timespec="seconds")
            updated = TreeMetadata(
                id=meta.id,
                title=meta.title,
                summary=summary,
                project_path=meta.project_path,
                transcript_path=str(self.current_path),
                created_at=meta.created_at,
                updated_at=now,
                parent_id=meta.parent_id,
                head_id=self._load_head_id(),
            )
            self._upsert_metadata(updated)
            return updated

    def rename_session(self, title: str) -> TreeMetadata | None:
        with self._lock:
            meta = self._metadata_for_path(self.current_path)
            if meta is None:
                return None
            now = datetime.now(UTC).isoformat(timespec="seconds")
            updated = TreeMetadata(
                id=meta.id,
                title=title,
                summary=meta.summary,
                project_path=meta.project_path,
                transcript_path=str(self.current_path),
                created_at=meta.created_at,
                updated_at=now,
                parent_id=meta.parent_id,
                head_id=self._load_head_id(),
            )
            self._upsert_metadata(updated)
            return updated

    def current_metadata(self) -> TreeMetadata | None:
        return self._metadata_for_path(self.current_path)

    def protocol_info(self) -> dict:
        return {
            "version": 1,
            "storage": "tree-jsonl-v1",
            "recovery_boundary": "head_id_and_entry_tree",
        }

    def get_tree(self) -> list[TreeNode]:
        entries = self.read_entries()
        if not entries:
            return []
        by_id = {e.id: e for e in entries}
        children: dict[str, list[SessionEntry]] = {}
        for e in entries:
            pid = e.parent_id
            if pid:
                children.setdefault(pid, []).append(e)
        head_id = self._load_head_id()
        heads: list[SessionEntry] = []
        if head_id and head_id in by_id:
            current = head_id
            while current and current in by_id:
                heads.insert(0, by_id[current])
                current = by_id[current].parent_id
        result: list[TreeNode] = []
        seen: set[str] = set()

        def walk(e: SessionEntry, depth: int) -> None:
            if e.id in seen:
                return
            seen.add(e.id)
            is_current = e.id == head_id
            type_label = e.type[:20]
            result.append(
                TreeNode(
                    id=e.id,
                    title=type_label,
                    depth=depth,
                    is_current=is_current,
                    is_leaf=e.id not in children,
                )
            )
            for child in sorted(
                children.get(e.id, []), key=lambda x: x.created_at
            ):
                walk(child, depth + 1)

        if heads:
            for h in heads:
                walk(h, 0)
        else:
            for e in entries:
                if e.parent_id is None:
                    walk(e, 0)
        return result

    @staticmethod
    def is_meaningful_session(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    model = TreeEntryModel.model_validate(data)
                except (ValidationError, json.JSONDecodeError):
                    continue
                if model.type in {"user", "assistant", "event"}:
                    return True
        except (OSError, json.JSONDecodeError):
            return False
        return False

    def find_latest_for_project(self, project_root: Path) -> SessionInfoView | None:
        resolved = str(project_root.resolve())
        for item in self.list_infos(limit=None):
            if item.project_path != resolved:
                continue
            if not self.is_meaningful_session(item.path):
                continue
            return item
        return None

    def find_by_id(self, session_id: str) -> SessionInfoView | None:
        try:
            safe_id = self._validate_session_id(session_id)
        except ValueError:
            return None
        candidate = self.sessions_dir / f"session-{safe_id}.jsonl"
        if candidate.exists() and candidate.is_file():
            meta = self._metadata_for_path(candidate)
            return self._view_for_path(candidate, meta)
        sessions_root = self.sessions_dir.resolve()
        for item in self._load_metadata():
            if item.id != safe_id:
                continue
            stored = Path(item.transcript_path)
            if not stored.is_absolute():
                stored = self.index_path.parent / stored
            stored_resolved = stored.resolve()
            try:
                stored_resolved.relative_to(sessions_root)
            except ValueError:
                continue
            if not stored_resolved.is_file():
                continue
            if (
                not stored_resolved.name.startswith("session-")
                or stored_resolved.suffix != ".jsonl"
            ):
                continue
            return self._view_for_path(stored_resolved, item)
        return None

    # ── 内部 ──

    def _load_head_id(self) -> str | None:
        meta = self._metadata_for_path(self.current_path)
        if meta is not None:
            return meta.head_id
        return None

    def _save_head_id(self, entry_id: str) -> None:
        meta = self._metadata_for_path(self.current_path)
        if meta is None:
            return
        now = datetime.now(UTC).isoformat(timespec="seconds")
        updated = TreeMetadata(
            id=meta.id,
            title=meta.title,
            summary=meta.summary,
            project_path=meta.project_path,
            transcript_path=str(self.current_path),
            created_at=meta.created_at,
            updated_at=now,
            parent_id=meta.parent_id,
            head_id=entry_id,
        )
        self._upsert_metadata(updated)

    def _session_paths(self) -> list[Path]:
        return sorted(
            self.sessions_dir.glob("session-*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _new_path(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = self.sessions_dir / f"session-{stamp}.jsonl"
        suffix = 1
        while path.exists():
            path = self.sessions_dir / f"session-{stamp}-{suffix}.jsonl"
            suffix += 1
        return path

    def _resolve_target(self, target: Path | str) -> Path:
        if isinstance(target, Path):
            return target
        text = target.strip()
        if not text:
            raise ValueError("empty session id")
        path = Path(text)
        if path.exists():
            return path
        for view in self.list_infos(limit=1000):
            if text in {view.id, view.title}:
                return view.path
        candidate = self.sessions_dir / f"session-{text}.jsonl"
        return candidate

    def _load_metadata(self) -> list[TreeMetadata]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_items = data.get("sessions", []) if isinstance(data, dict) else []
        items = []
        for raw in raw_items:
            try:
                metadata = TreeMetadata.model_validate(raw)
            except ValidationError:
                logging.warning("skipping malformed session metadata: %s", raw)
                continue
            items.append(metadata)
        return items

    def _write_metadata(self, items: list[TreeMetadata]) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "storage": "tree-jsonl-v1",
                "sessions": [item.model_dump() for item in items],
            }
            self.index_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _upsert_metadata(self, metadata: TreeMetadata) -> None:
        items = [item for item in self._load_metadata() if item.id != metadata.id]
        items.insert(0, metadata)
        self._write_metadata(items)

    def _metadata_for_path(self, path: Path) -> TreeMetadata | None:
        sid = self._session_id(path)
        for item in self._load_metadata():
            if item.id == sid:
                return item
        return None

    def _view_for_path(
        self, path: Path, metadata: TreeMetadata | None
    ) -> SessionInfoView:
        if metadata is None:
            stat = path.stat()
            updated = datetime.fromtimestamp(stat.st_mtime).isoformat(
                timespec="seconds"
            )
            sid = self._session_id(path)
            return SessionInfoView(
                id=sid,
                title=f"Session {sid}",
                summary="No summary available.",
                updated_at=updated,
                path=path,
            )
        return SessionInfoView(
            id=metadata.id,
            title=metadata.title,
            summary=metadata.summary,
            updated_at=metadata.updated_at,
            path=path,
            project_path=metadata.project_path,
            parent_id=metadata.parent_id,
        )

    @staticmethod
    def _session_id(path: Path) -> str:
        name = path.stem
        return name.removeprefix("session-")

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not session_id:
            raise ValueError("session id is empty")
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"invalid session id: {session_id!r}")
        return session_id


def _make_title(text: str | None) -> str:
    cleaned = _collapse_text(text or "")
    if not cleaned:
        return "Untitled conversation"
    return _truncate(cleaned, 72)


def _make_initial_summary(text: str | None) -> str:
    cleaned = _collapse_text(text or "")
    if not cleaned:
        return "Conversation started."
    return f"First request: {_truncate(cleaned, SUMMARY_TITLE_CHARS)}"


def _make_conversation_summary(user: str, assistant: str | None) -> str:
    user_text = _truncate(_collapse_text(user), SUMMARY_USER_CHARS)
    if not assistant:
        return f"First request: {user_text}"
    assistant_text = _truncate(_collapse_text(assistant), SUMMARY_ASSISTANT_CHARS)
    return f"First request: {user_text} Answer preview: {assistant_text}"


def _collapse_text(text: str) -> str:
    return " ".join(str(text).split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
