"""长任务运行编排状态独立存储。

将 orchestration state（lease、retry、attempt）从 ``task.payload`` 分离到
``.local/orchestration/{task_id}.json``，避免 agent 误改控制面，并让
``save_progress`` 可以安全地 merge payload 而不覆盖运行时状态。

lease 索引 ``.local/orchestration/.lease_index`` 按 task_id -> expires_at
建索引，``list_expired`` 仅扫描索引而非全表 task 文件。
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import filelock

logger = logging.getLogger("xcode.harness.orchestration_store")


@dataclass(frozen=True)
class TaskRunState:
    """长任务运行编排状态。"""

    task_id: int
    status: str
    attempt: int
    retry_limit: int
    lease_expires_at: str
    subtask_ids: list[int]


class OrchestrationStore:
    """orchestration state 的文件系统真值源，独立于 TaskStore.payload。"""

    def __init__(self, root: Path, lock_timeout_seconds: float = 5.0) -> None:
        self.root = root
        self.dir = root / ".local" / "orchestration"
        self.lease_index_path = self.dir / ".lease_index"
        self.lock_timeout_seconds = lock_timeout_seconds

    def _state_path(self, task_id: int) -> Path:
        return self.dir / f"{task_id}.json"

    def _lock_path(self) -> Path:
        return self.dir / ".lock_file"

    def _lock(self) -> filelock.FileLock:
        self.dir.mkdir(parents=True, exist_ok=True)
        return filelock.FileLock(self._lock_path(), timeout=self.lock_timeout_seconds)

    def get(self, task_id: int) -> TaskRunState | None:
        """读取 orchestration 状态；不存在返回 None。"""
        path = self._state_path(int(task_id))
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("failed to read orchestration state for task %s", task_id)
            return None
        return _decode_state(data)

    def set(self, state: TaskRunState) -> None:
        """写入 orchestration 状态并更新 lease 索引。"""
        with self._lock():
            self._write_state(state)
            self._update_lease_index({str(state.task_id): state.lease_expires_at})

    def delete(self, task_id: int) -> None:
        """删除 orchestration 状态并从 lease 索引移除。"""
        with self._lock():
            path = self._state_path(int(task_id))
            if path.exists():
                path.unlink()
            self._update_lease_index({}, remove={str(int(task_id))})

    def list_expired(self, now: float | None = None) -> list[int]:
        """返回 lease 已过期的 task_id 列表（按索引查询，不扫全表）。

        索引损坏时回退到全目录扫描并重建索引。
        """
        now = now if now is not None else time.time()
        with self._lock():
            index = self._read_lease_index()
            if index is None:
                logger.warning(
                    "lease index corrupted or missing, falling back to full scan"
                )
                return self._full_scan_expired(now)
            expired_ids: list[int] = []
            for task_id_str, expires_at in index.items():
                expires_epoch = _parse_timestamp(str(expires_at))
                if expires_epoch <= 0:
                    continue
                if expires_epoch < now:
                    try:
                        expired_ids.append(int(task_id_str))
                    except ValueError:
                        continue
            return expired_ids

    def list_active(self) -> list[TaskRunState]:
        """返回所有已注册的 orchestration 状态。"""
        with self._lock():
            states: list[TaskRunState] = []
            for path in self.dir.glob("*.json"):
                if path.name.startswith("."):
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    states.append(_decode_state(data))
                except (json.JSONDecodeError, OSError):
                    continue
            return sorted(states, key=lambda s: s.task_id)

    def _write_state(self, state: TaskRunState) -> None:
        path = self._state_path(state.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    def _read_lease_index(self) -> dict[str, str] | None:
        """读取 lease 索引；损坏返回 None 触发回退。"""
        if not self.lease_index_path.exists():
            return {}
        try:
            data = json.loads(self.lease_index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        result: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str):
                result[k] = v
        return result

    def _update_lease_index(
        self,
        updates: dict[str, str],
        *,
        remove: set[str] | None = None,
    ) -> None:
        """合并更新 lease 索引并原子写回。调用方必须持锁。"""
        current = self._read_lease_index() or {}
        if remove:
            for key in remove:
                current.pop(key, None)
        current.update(updates)
        self._atomic_write_json(self.lease_index_path, current)

    def _full_scan_expired(self, now: float) -> list[int]:
        """全目录扫描过期的 task_id，并重建索引。"""
        expired: list[int] = []
        index: dict[str, str] = {}
        for path in self.dir.glob("*.json"):
            if path.name.startswith("."):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            task_id = int(data.get("task_id", 0))
            expires_at = str(data.get("lease_expires_at", ""))
            if task_id and expires_at:
                index[str(task_id)] = expires_at
                if _parse_timestamp(expires_at) < now:
                    expired.append(task_id)
        self._atomic_write_json(self.lease_index_path, index)
        return sorted(expired)

    def _atomic_write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, path)


def _decode_state(data: dict[str, Any]) -> TaskRunState:
    subtask_ids = data.get("subtask_ids") or []
    if not isinstance(subtask_ids, list):
        subtask_ids = []
    clean_ids: list[int] = []
    for value in subtask_ids:
        if isinstance(value, int):
            clean_ids.append(value)
        elif str(value).isdigit():
            clean_ids.append(int(value))
    return TaskRunState(
        task_id=int(data.get("task_id", 0)),
        status=str(data.get("status", "")),
        attempt=int(data.get("attempt", 0)),
        retry_limit=int(data.get("retry_limit", 0)),
        lease_expires_at=str(data.get("lease_expires_at", "")),
        subtask_ids=clean_ids,
    )


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return 0.0
