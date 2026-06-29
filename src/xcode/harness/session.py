from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, UTC
import filelock
import json
import logging
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
import shutil
from collections.abc import Callable

from xcode.harness.skill_activation import is_skill_activation_content

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


# 会话 id 的白名单正则：只允许字母数字、连字符和星号
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# 会话摘要字符限制（终端显示和可读性优化）
SUMMARY_USER_CHARS = 120  # 用户消息摘要：约一行终端显示
SUMMARY_ASSISTANT_CHARS = 180  # 助手消息摘要：约一行半终端显示
SUMMARY_TITLE_CHARS = 160  # 会话标题：适配文件浏览器列宽度

# 会话存储协议版本
SESSION_INDEX_VERSION = 1
SESSION_STORAGE_PROTOCOL = "jsonl-v1"
SESSION_RECOVERY_BOUNDARY = "current_transcript_and_session_tree"


FORK_TYPES = frozenset(["explore", "verify", "isolate"])


class SessionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    content: JsonValue
    created_at: str


class SessionMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    summary: str
    project_path: str
    transcript_path: str
    created_at: str
    updated_at: str
    parent_id: str | None = None
    fork_type: str | None = None

    @field_validator("fork_type")
    @classmethod
    def _validate_fork_type(cls, v: str | None) -> str | None:
        if v is not None and v not in FORK_TYPES:
            return None
        return v


class SessionProtocolInfo(BaseModel):
    """会话持久化协议描述。"""

    model_config = ConfigDict(frozen=True)

    version: int
    storage: str
    recovery_boundary: str


class SessionStore:
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

    def append(self, record_type: str, content: JsonValue) -> None:
        with self._lock:
            record = SessionRecord(
                type=record_type,
                content=content,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            with self.current_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            if record_type == "user":
                self.ensure_metadata(str(content))

    def clear(self) -> None:
        with self._lock:
            self.current_path = self._new_path()

    def _fork_base(
        self,
        fork_type: str | None,
        make_title: Callable[[SessionMetadata], str],
        make_summary: Callable[[SessionMetadata], str],
        copy_transcript: bool,
    ) -> SessionMetadata:
        if fork_type is not None and fork_type not in FORK_TYPES:
            raise ValueError(
                f"fork_type must be one of {FORK_TYPES}, got {fork_type!r}"
            )
        with self._lock:
            parent = self.ensure_metadata()
            fork_path = self._new_path()
            if copy_transcript and self.current_path.exists():
                shutil.copy2(self.current_path, fork_path)
            now = datetime.now(UTC).isoformat(timespec="seconds")
            meta = SessionMetadata(
                id=self._session_id(fork_path),
                title=make_title(parent),
                summary=make_summary(parent),
                project_path=parent.project_path,
                transcript_path=self._relative_transcript_path(fork_path),
                created_at=now,
                updated_at=now,
                parent_id=parent.id,
                fork_type=fork_type,
            )
            self._upsert_metadata(meta)
            self.current_path = fork_path
            return meta

    def fork_into(self, fork_type: str | None = None) -> SessionMetadata:
        return self._fork_base(
            fork_type=fork_type,
            make_title=lambda p: f"Fork of {p.title}",
            make_summary=lambda p: p.summary,
            copy_transcript=True,
        )

    def fork_clean_into(
        self, fork_type: str | None = None, title: str | None = None
    ) -> SessionMetadata:
        return self._fork_base(
            fork_type=fork_type,
            make_title=lambda p: title or f"Clean Fork of {p.title}",
            make_summary=lambda _: "Conversation started (clean fork).",
            copy_transcript=False,
        )

    def load_records(self, path: Path | None = None) -> list[SessionRecord]:
        target = path or self.current_path
        if not target.exists():
            return []
        records = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                try:
                    record = SessionRecord.model_validate(data)
                except ValidationError:
                    continue
                records.append(record)
        return records

    def resume(self, target: Path | str) -> None:
        path = self._resolve_resume_target(target)
        if not path.exists():
            raise ValueError(f"session does not exist: {path}")
        self.current_path = path

    def switch_branch(self, target: str) -> SessionMetadataView:
        """切换到会话树中的指定分支。"""
        self.resume(target)
        return self._view_for_path(self.current_path, self.current_metadata())

    def resume_latest(self) -> Path | None:
        sessions = [item.path for item in self.list_session_infos(limit=1)]
        if not sessions:
            return None
        self.current_path = sessions[0]
        return self.current_path

    def rewind_turns(self, turns: int = 1) -> int:
        with self._lock:
            records = self.load_records()
            if turns <= 0 or not records:
                return 0
            user_indices = [
                index for index, record in enumerate(records) if record.type == "user"
            ]
            if not user_indices:
                return 0
            keep_until = user_indices[max(0, len(user_indices) - turns)]
            kept = records[:keep_until]
            with self.current_path.open("w", encoding="utf-8") as handle:
                for record in kept:
                    handle.write(
                        json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
                    )
            return len(records) - len(kept)

    def user_turn_count(self) -> int:
        """返回当前 transcript 中保留的用户轮次数。"""
        return sum(1 for record in self.load_records() if record.type == "user")

    def compact_current_session(self, max_tool_result_chars: int = 200) -> int:
        """压缩当前会话事件日志，截断过长的工具执行结果内容。"""
        with self._lock:
            records = self.load_records()
            if not records:
                return 0
            compacted_count = 0
            new_records = []
            for record in records:
                if record.type == "event" and isinstance(record.content, dict):
                    if record.content.get("type") == "tool_result":
                        data = record.content.get("data")
                        if isinstance(data, dict) and "content" in data:
                            content_str = str(data["content"])
                            if is_skill_activation_content(content_str):
                                new_records.append(record)
                                continue
                            if len(content_str) > max_tool_result_chars:
                                original_len = len(content_str)
                                data["content"] = (
                                    "[Previous tool_result compacted; "
                                    f"{original_len} chars removed]"
                                )
                                compacted_count += 1
                new_records.append(record)
            if compacted_count > 0:
                with self.current_path.open("w", encoding="utf-8") as handle:
                    for record in new_records:
                        handle.write(
                            json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
                        )
            return compacted_count

    def list_sessions(self, limit: int = 10) -> list[Path]:
        return [item.path for item in self.list_session_infos(limit=limit)]

    def list_session_infos(self, limit: int | None = 10) -> list[SessionMetadataView]:
        known = {item.id: item for item in self._load_metadata()}
        views: list[SessionMetadataView] = []
        for path in self._session_paths():
            metadata = known.get(self._session_id(path))
            views.append(self._view_for_path(path, metadata))
        sorted_views = sorted(views, key=lambda item: item.updated_at, reverse=True)
        if limit is not None:
            return sorted_views[:limit]
        return sorted_views

    def ensure_metadata(self, first_user_text: str | None = None) -> SessionMetadata:
        existing = self._metadata_for_path(self.current_path)
        if existing is not None:
            return existing
        now = datetime.now(UTC).isoformat(timespec="seconds")
        title = (
            _make_title(first_user_text) if first_user_text else "Untitled conversation"
        )
        summary = _make_initial_summary(first_user_text)
        metadata = SessionMetadata(
            id=self._session_id(self.current_path),
            title=title,
            summary=summary,
            project_path=str(self.project_root),
            transcript_path=self._relative_transcript_path(self.current_path),
            created_at=now,
            updated_at=now,
        )
        self._upsert_metadata(metadata)
        return metadata

    def update_summary(self) -> SessionMetadata | None:
        records = self.load_records()
        user = next(
            (str(record.content) for record in records if record.type == "user"), None
        )
        if user is None:
            return None
        assistant = next(
            (str(record.content) for record in records if record.type == "assistant"),
            None,
        )
        current = self.ensure_metadata(user)
        metadata = SessionMetadata(
            id=current.id,
            title=current.title or _make_title(user),
            summary=_make_conversation_summary(user, assistant),
            project_path=current.project_path,
            transcript_path=current.transcript_path,
            created_at=current.created_at,
            updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._upsert_metadata(metadata)
        return metadata

    def rename_session(self, title: str) -> SessionMetadata | None:
        """重命名当前会话。"""
        current = self.current_metadata()
        if current is None:
            return None
        metadata = SessionMetadata(
            id=current.id,
            title=title,
            summary=current.summary,
            project_path=current.project_path,
            transcript_path=current.transcript_path,
            created_at=current.created_at,
            updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            parent_id=current.parent_id,
            fork_type=current.fork_type,
        )
        self._upsert_metadata(metadata)
        return metadata

    def current_metadata(self) -> SessionMetadata | None:
        return self._metadata_for_path(self.current_path)

    def protocol_info(self) -> SessionProtocolInfo:
        """返回当前会话索引使用的持久化协议。"""
        return SessionProtocolInfo(
            version=SESSION_INDEX_VERSION,
            storage=SESSION_STORAGE_PROTOCOL,
            recovery_boundary=SESSION_RECOVERY_BOUNDARY,
        )

    def get_tree(self) -> list[TreeNode]:
        """以当前会话为视角，构建会话树：祖先链 + 当前子树。

        返回按深度排序的 TreeNode 列表。
        """
        all_meta = {m.id: m for m in self._load_metadata()}
        current_id = self._session_id(self.current_path)

        # 构建父子索引
        children: dict[str, list[SessionMetadata]] = {}
        for m in all_meta.values():
            pid = m.parent_id
            if pid:
                children.setdefault(pid, []).append(m)

        result: list[TreeNode] = []

        # 从根节点开始构建
        current = all_meta.get(current_id)
        if current is None:
            return result

        # 先构建祖先链（root→parent→current）
        chain: list[SessionMetadata] = [current]
        while chain[-1].parent_id and chain[-1].parent_id in all_meta:
            chain.append(all_meta[chain[-1].parent_id])
        chain.reverse()

        seen: set[str] = set()

        def walk(meta: SessionMetadata, depth: int) -> None:
            if meta.id in seen:
                return
            seen.add(meta.id)
            is_current = meta.id == current_id
            result.append(
                TreeNode(
                    id=meta.id,
                    title=meta.title,
                    fork_type=meta.fork_type,
                    depth=depth,
                    is_current=is_current,
                    is_leaf=meta.id not in children,
                )
            )
            for child in sorted(children.get(meta.id, []), key=lambda m: m.created_at):
                walk(child, depth + 1)

        for ancestor in chain:
            walk(ancestor, 0 if ancestor.id == chain[0].id else chain.index(ancestor))

        return result

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

    def _resolve_resume_target(self, target: Path | str) -> Path:
        if isinstance(target, Path):
            return target
        text = target.strip()
        if not text:
            raise ValueError("empty session id")
        path = Path(text)
        if path.exists():
            return path
        for view in self.list_session_infos(limit=1000):
            if text in {view.id, view.title}:
                return view.path
        candidate = self.sessions_dir / f"session-{text}.jsonl"
        return candidate

    def _load_metadata(self) -> list[SessionMetadata]:
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
                metadata = SessionMetadata.model_validate(raw)
            except ValidationError:
                logging.warning("skipping malformed session metadata: %s", raw)
                continue
            items.append(metadata)
        return items

    def _write_metadata(self, items: list[SessionMetadata]) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            protocol = self.protocol_info()
            payload = {
                "version": protocol.version,
                "storage": protocol.storage,
                "recovery_boundary": protocol.recovery_boundary,
                "sessions": [item.model_dump() for item in items],
            }
            self.index_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _upsert_metadata(self, metadata: SessionMetadata) -> None:
        items = [item for item in self._load_metadata() if item.id != metadata.id]
        items.insert(0, metadata)
        self._write_metadata(items)

    def _metadata_for_path(self, path: Path) -> SessionMetadata | None:
        session_id = self._session_id(path)
        for item in self._load_metadata():
            if item.id == session_id:
                return item
        return None

    def _view_for_path(
        self,
        path: Path,
        metadata: SessionMetadata | None,
    ) -> SessionMetadataView:
        if metadata is None:
            stat = path.stat()
            updated = datetime.fromtimestamp(stat.st_mtime).isoformat(
                timespec="seconds"
            )
            session_id = self._session_id(path)
            return SessionMetadataView(
                id=session_id,
                title=f"Session {session_id}",
                summary="No summary available.",
                updated_at=updated,
                path=path,
            )
        return SessionMetadataView(
            id=metadata.id,
            title=metadata.title,
            summary=metadata.summary,
            updated_at=metadata.updated_at,
            path=path,
            project_path=metadata.project_path,
            parent_id=metadata.parent_id,
            fork_type=metadata.fork_type,
        )

    def _relative_transcript_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.index_path.parent))
        except ValueError:
            return os.path.relpath(path, self.index_path.parent)

    @staticmethod
    def _session_id(path: Path) -> str:
        name = path.stem
        return name.removeprefix("session-")

    @property
    def session_id(self) -> str:
        """返回当前会话的 logical_session_id。"""
        return self._session_id(self.current_path)

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        """Whitelist-based session id validation. Raises ValueError for invalid input."""
        if not session_id:
            raise ValueError("session id is empty")
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"invalid session id: {session_id!r}")
        return session_id

    @staticmethod
    def is_meaningful_session(path: Path) -> bool:
        """True if the session file contains at least one real transcript record."""
        if not path.exists():
            return False
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                try:
                    record = SessionRecord.model_validate(data)
                except ValidationError:
                    continue
                if record.type in {"user", "assistant", "event"}:
                    return True
        except (OSError, json.JSONDecodeError):
            return False
        return False

    def find_latest_for_project(self, project_root: Path) -> SessionMetadataView | None:
        """返回当前项目最新的有意义会话，或 None。

        不调用 self.resume()。调用者必须使用 store.resume(view.id)。
        如果当前会话有意义且是最新的，会被包含在内。
        """
        resolved = str(project_root.resolve())
        for item in self.list_session_infos(limit=None):
            if item.project_path != resolved:
                continue
            if not self.is_meaningful_session(item.path):
                continue
            return item
        return None

    def find_by_id(self, session_id: str) -> SessionMetadataView | None:
        """Exact session lookup by id. Not capped by list_session_infos limit."""
        try:
            safe_id = self._validate_session_id(session_id)
        except ValueError:
            return None
        # direct path construction (safe after whitelist validation)
        candidate = self.sessions_dir / f"session-{safe_id}.jsonl"
        if candidate.exists() and candidate.is_file():
            metadata = self._metadata_for_path(candidate)
            return self._view_for_path(candidate, metadata)
        # metadata fallback — validate path containment via relative_to
        sessions_root = self.sessions_dir.resolve()
        for item in self._load_metadata():
            if item.id != safe_id:
                continue
            stored = self.index_path.parent / item.transcript_path
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


@dataclass(frozen=True)
class SessionMetadataView:
    id: str
    title: str
    summary: str
    updated_at: str
    path: Path
    project_path: str = ""
    parent_id: str | None = None
    fork_type: str | None = None


def _make_title(text: str | None) -> str:
    """从首条消息生成会话标题，最长 72 字符。"""
    cleaned = _collapse_text(text or "")
    if not cleaned:
        return "Untitled conversation"
    return _truncate(cleaned, 72)


def _make_initial_summary(text: str | None) -> str:
    """从首条消息生成初始摘要。"""
    cleaned = _collapse_text(text or "")
    if not cleaned:
        return "Conversation started."
    return f"First request: {_truncate(cleaned, SUMMARY_TITLE_CHARS)}"


def _make_conversation_summary(user: str, assistant: str | None) -> str:
    """从首条用户消息和助手响应生成会话摘要。"""
    user_text = _truncate(_collapse_text(user), SUMMARY_USER_CHARS)
    if not assistant:
        return f"First request: {user_text}"
    assistant_text = _truncate(_collapse_text(assistant), SUMMARY_ASSISTANT_CHARS)
    return f"First request: {user_text} Answer preview: {assistant_text}"


def _collapse_text(text: str) -> str:
    """折叠空白字符：连续空白替换为单个空格，删除首尾空白。"""
    return " ".join(str(text).split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


@dataclass(frozen=True)
class TreeNode:
    id: str
    title: str
    fork_type: str | None
    depth: int
    is_current: bool
    is_leaf: bool
