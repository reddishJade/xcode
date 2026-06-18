from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
import json
import logging
from pathlib import Path
import subprocess
import threading
from typing import Literal

logger = logging.getLogger(__name__)

MAX_SNAPSHOT_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

SNAPSHOT_EXCLUDES: list[str] = [
    ".git/",
    ".local/",
    "node_modules/",
    "vendor/",
    "__pycache__/",
    "*.pyc",
    "dist/",
    "build/",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff*",
    "*.eot",
    "*.ttf",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.zip",
    "*.tar*",
    "*.gz",
]


class SnapshotUnsupportedError(RuntimeError):
    """本步骤仅支持 git 工程。"""


@dataclass
class SkippedFileInfo:
    path: str
    reason: str


@dataclass
class ChangeEntry:
    path: str
    kind: Literal["modified", "created", "deleted"]


@dataclass
class TurnSnapshotRecord:
    turn_id: str
    pre_snapshot_id: str
    post_snapshot_id: str
    changed_files: list[ChangeEntry]
    skipped_files: list[SkippedFileInfo] = field(default_factory=list)
    timestamp: str = ""
    undone: bool = False
    tool_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "pre_snapshot_id": self.pre_snapshot_id,
            "post_snapshot_id": self.post_snapshot_id,
            "changed_files": [
                {"path": c.path, "kind": c.kind} for c in self.changed_files
            ],
            "skipped_files": [
                {"path": s.path, "reason": s.reason} for s in self.skipped_files
            ],
            "timestamp": self.timestamp,
            "undone": self.undone,
            "tool_names": self.tool_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TurnSnapshotRecord:
        raw_changed = data.get("changed_files", [])
        raw_skipped = data.get("skipped_files", [])
        if not isinstance(raw_changed, list):
            raw_changed = []
        if not isinstance(raw_skipped, list):
            raw_skipped = []
        changed: list[ChangeEntry] = []
        for c in raw_changed:
            if not isinstance(c, dict):
                continue
            path = str(c.get("path", ""))
            raw_kind = str(c.get("kind", "modified"))
            kind: Literal["modified", "created", "deleted"] = "modified"
            if raw_kind == "created":
                kind = "created"
            elif raw_kind == "deleted":
                kind = "deleted"
            changed.append(ChangeEntry(path=path, kind=kind))
        return cls(
            turn_id=str(data.get("turn_id", "")),
            pre_snapshot_id=str(data.get("pre_snapshot_id", "")),
            post_snapshot_id=str(data.get("post_snapshot_id", "")),
            changed_files=changed,
            skipped_files=[
                SkippedFileInfo(
                    path=str(s.get("path", "")),
                    reason=str(s.get("reason", "")),
                )
                for s in raw_skipped
                if isinstance(s, dict)
            ],
            timestamp=str(data.get("timestamp", "")),
            undone=bool(data.get("undone", False)),
            tool_names=_tool_names_from_data(data.get("tool_names")),
        )


@dataclass
class SnapshotResult:
    snapshot_id: str
    skipped_files: list[SkippedFileInfo]


class SnapshotService:
    """基于隐藏 git tree 的预/后拍快照服务。

    每个 git 命令显式使用 --git-dir <hidden> --work-tree <project_root>，
    从不碰用户 .git/、index、stash、HEAD 或 ref。
    snapshot_id 是 git tree hash（git write-tree），不是 commit hash。
    """

    def __init__(self, project_root: Path, session_id: str) -> None:
        self._project_root = project_root.resolve()
        self._git_dir = (
            project_root / ".local" / "snapshots" / session_id / ".git"
        ).resolve()
        self._lock = threading.Lock()
        self._skipped: list[SkippedFileInfo] = []
        self._init_repo()

    def _git(
        self,
        args: list[str],
        check: bool = True,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        cmd = [
            "git",
            "--git-dir",
            str(self._git_dir),
            "--work-tree",
            str(self._project_root),
            *args,
        ]
        return subprocess.run(
            cmd,
            cwd=self._project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def _init_repo(self) -> None:
        if (self._git_dir / "HEAD").exists():
            return
        self._git_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git(["init"])
        self._git(["config", "core.worktree", str(self._project_root)])
        exclude_file = self._git_dir / "info" / "exclude"
        exclude_file.write_text("\n".join(SNAPSHOT_EXCLUDES) + "\n")

    def _record_skipped(self, path: str, reason: str) -> None:
        self._skipped.append(SkippedFileInfo(path=path, reason=reason))

    def _pop_skipped(self) -> list[SkippedFileInfo]:
        items = list(self._skipped)
        self._skipped.clear()
        return items

    def _is_env_secret(self, rel_path: str) -> bool:
        name = Path(rel_path).name
        if name == ".env.example":
            return False
        if name == ".env" or name.startswith(".env."):
            return True
        return False

    def _enumerate_files(self) -> list[str]:
        output = self._git(
            [
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--full-name",
            ]
        )
        candidates = [
            line.strip() for line in output.stdout.splitlines() if line.strip()
        ]
        candidates.sort()

        result: list[str] = []
        for rel_path in candidates:
            if self._is_env_secret(rel_path):
                self._record_skipped(rel_path, "excluded: environment secret file")
                continue
            abs_path = (self._project_root / rel_path).resolve()
            try:
                size = abs_path.stat().st_size
            except OSError:
                self._record_skipped(rel_path, "skipped: stat error")
                continue
            if size > MAX_SNAPSHOT_FILE_BYTES:
                self._record_skipped(
                    rel_path,
                    f"skipped: file too large ({size} > {MAX_SNAPSHOT_FILE_BYTES} bytes)",
                )
                continue
            result.append(rel_path)
        return result

    def track(self) -> SnapshotResult:
        with self._lock:
            self._git(["read-tree", "--empty"])
            for rel_path in self._enumerate_files():
                self._git(["add", rel_path])
            result = self._git(["write-tree"])
            snapshot_id = result.stdout.strip()
            skipped = self._pop_skipped()
            return SnapshotResult(
                snapshot_id=snapshot_id,
                skipped_files=skipped,
            )

    def diff(self, pre_tree: str, post_tree: str) -> list[ChangeEntry]:
        with self._lock:
            result = self._git(
                [
                    "diff-tree",
                    "--name-status",
                    "-r",
                    "--no-renames",
                    pre_tree,
                    post_tree,
                ]
            )
        entries: list[ChangeEntry] = []
        status_map: dict[str, Literal["modified", "created", "deleted"]] = {
            "M": "modified",
            "A": "created",
            "D": "deleted",
        }
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or len(line) < 2:
                continue
            status_char = line[0]
            path_part = line[1:].strip()
            kind = status_map.get(status_char)
            if kind:
                entries.append(ChangeEntry(path=path_part, kind=kind))
        return entries

    def _validate_path(self, rel_path: str) -> str:
        path = rel_path.replace("\\", "/").strip()
        if not path:
            raise ValueError("empty path")
        if path.startswith("/"):
            raise ValueError(f"absolute path not allowed: {path}")
        if ".." in path.split("/"):
            raise ValueError(f"parent-directory traversal not allowed: {path}")
        candidate = (self._project_root / path).resolve()
        try:
            candidate.relative_to(self._project_root)
        except ValueError:
            raise ValueError(f"path escapes project root: {path}")
        return path

    def has_conflict(self, post_tree: str, rel_path: str) -> bool:
        result = self._git(
            ["diff", "--exit-code", post_tree, "--", rel_path],
            check=False,
        )
        return result.returncode != 0

    def restore_file(self, snapshot_id: str, rel_path: str) -> None:
        self._validate_path(rel_path)
        self._git(["checkout", snapshot_id, "--", rel_path])


def _tool_names_from_data(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


class SnapshotStore:
    """快照存储管理器，维护 TurnSnapshotRecord 索引。"""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        if not self._is_git_available(self._project_root):
            raise SnapshotUnsupportedError("Snapshot undo requires a git repository.")
        self._lock = threading.Lock()
        self._services: dict[str, SnapshotService] = {}

    @staticmethod
    def _is_git_available(project_root: Path) -> bool:
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                timeout=2,
                check=True,
            )
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=project_root,
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False

    @property
    def project_root(self) -> Path:
        return self._project_root

    def service(self, session_id: str) -> SnapshotService:
        if session_id not in self._services:
            self._services[session_id] = SnapshotService(self._project_root, session_id)
        return self._services[session_id]

    def _index_path(self, session_id: str) -> Path:
        return self._project_root / ".local" / "snapshots" / session_id / "index.json"

    def _lock_path(self, session_id: str) -> Path:
        return self._project_root / ".local" / "snapshots" / session_id / "index.lock"

    def next_turn_id(self, session_id: str) -> str:
        records = self._load_index(session_id)
        if not records:
            return "001"
        last_id = max(int(r.turn_id) for r in records)
        return f"{last_id + 1:03d}"

    def _load_index(self, session_id: str) -> list[TurnSnapshotRecord]:
        path = self._index_path(session_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to load snapshot index: %s", path)
            return []
        raw_turns = data.get("turns", []) if isinstance(data, dict) else []
        return [
            TurnSnapshotRecord.from_dict(t) for t in raw_turns if isinstance(t, dict)
        ]

    def _write_index(self, session_id: str, records: list[TurnSnapshotRecord]) -> None:
        path = self._index_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "turns": [r.to_dict() for r in records],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def record_turn(
        self,
        session_id: str,
        turn_id: str,
        pre_snapshot_id: str,
        post_snapshot_id: str,
        changed_files: list[ChangeEntry],
        skipped_files: list[SkippedFileInfo] | None = None,
        tool_names: list[str] | None = None,
    ) -> None:
        with self._lock:
            records = self._load_index(session_id)
            record = TurnSnapshotRecord(
                turn_id=turn_id,
                pre_snapshot_id=pre_snapshot_id,
                post_snapshot_id=post_snapshot_id,
                changed_files=changed_files,
                skipped_files=skipped_files or [],
                timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                tool_names=tool_names or [],
            )
            records.append(record)
            self._write_index(session_id, records)

    def list_records(self, session_id: str) -> list[TurnSnapshotRecord]:
        return self._load_index(session_id)

    def get_undoable_records(self, session_id: str, n: int) -> list[TurnSnapshotRecord]:
        records = self._load_index(session_id)
        undoable = [r for r in records if not r.undone]
        if n <= 0:
            return []
        return undoable[-n:]

    def update_record(self, session_id: str, record: TurnSnapshotRecord) -> None:
        with self._lock:
            records = self._load_index(session_id)
            for i, r in enumerate(records):
                if r.turn_id == record.turn_id:
                    records[i] = record
                    break
            self._write_index(session_id, records)
