"""实验性轻量任务存储和任务工具。

每个任务保存为 `.local/tasks.json.d/{id}.json`，`.highwatermark` 记录已经分配
过的最大顺序 ID。目录锁用于保护 ID 分配、更新和领取，避免并发写入互相覆盖。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from xcode.harness.skills import ToolInput, ToolSpec


PENDING = "pending"
CLAIMED = "claimed"
COMPLETED = "completed"
VALID_STATUSES = (PENDING, CLAIMED, COMPLETED)


class ConcurrentModificationError(RuntimeError):
    """乐观锁版本冲突：尝试写入的版本不是最新版本。"""


CREATE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Title of the task"},
        "blocked_by": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "IDs of tasks that this task is blocked by",
        },
    },
    "required": ["title"],
    "additionalProperties": False,
}

UPDATE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
            "description": "ID of the task to update",
        },
        "status": {
            "type": "string",
            "enum": [PENDING, CLAIMED, COMPLETED],
            "description": "New status of the task",
        },
        "title": {"type": "string", "description": "New title of the task"},
        "blocked_by": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "IDs of tasks that this task is blocked by",
        },
        "expected_version": {
            "type": "integer",
            "description": "Optimistic lock: fail if current version does not match",
        },
    },
    "required": ["id"],
    "additionalProperties": False,
}

CLAIM_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {
            "type": "integer",
            "description": "ID of the task to claim",
        },
        "claimant": {
            "type": "string",
            "description": "Identifier of the agent claiming the task",
        },
    },
    "required": ["task_id", "claimant"],
    "additionalProperties": False,
}

LIST_TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "view": {
            "type": "string",
            "enum": ["kanban", "topological", "raw"],
            "description": "View style to render the tasks",
        }
    },
    "additionalProperties": False,
}

GET_TASK_SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "integer", "description": "Task ID"}},
    "required": ["id"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TaskRecord:
    id: int
    title: str
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    version: int = 1
    claimed_by: str | None = None
    claimed_at: str | None = None


class TaskStore:
    def __init__(self, root: Path, lock_timeout_seconds: float = 5.0) -> None:
        self.root = root
        self.tasks_dir = root / ".local" / "tasks.json.d"
        self.highwatermark_path = self.tasks_dir / ".highwatermark"
        self.lock_dir = self.tasks_dir / ".lock"
        self.lock_timeout_seconds = lock_timeout_seconds

    def create(self, title: str, payload: dict[str, Any] | None = None) -> TaskRecord:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title is required")
        with self.locked():
            task_id = self._next_id()
            now = _timestamp()
            task = TaskRecord(
                id=task_id,
                title=clean_title,
                status=PENDING,
                payload=dict(payload or {}),
                created_at=now,
                updated_at=now,
            )
            self._write(task)
            return task

    def get(self, task_id: int | str) -> TaskRecord:
        path = self._task_path(task_id)
        if not path.exists():
            raise KeyError(f"unknown task: {task_id}")
        return _decode(path.read_text(encoding="utf-8"))

    def list(self) -> list[TaskRecord]:
        if not self.tasks_dir.exists():
            return []
        tasks = [
            _decode(path.read_text(encoding="utf-8"))
            for path in self.tasks_dir.glob("*.json")
            if path.name != ".highwatermark"
        ]
        return sorted(tasks, key=lambda task: task.id)

    def update(
        self,
        task_id: int | str,
        *,
        title: str | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> TaskRecord:
        with self.locked():
            return self._apply_update(
                task_id,
                title=title,
                status=status,
                payload=payload,
                expected_version=expected_version,
            )

    def _apply_update(
        self,
        task_id: int | str,
        *,
        title: str | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> TaskRecord:
        """不加锁的内部更新，调用方必须已持有 self.locked()。"""
        task = self.get(task_id)
        if expected_version is not None and task.version != expected_version:
            raise ConcurrentModificationError(
                f"task #{task.id} version mismatch: expected {expected_version}, got {task.version}"
            )
        new_title = task.title if title is None else title.strip()
        if not new_title:
            raise ValueError("title is required")
        if status is not None:
            clean_status = status.strip()
            if clean_status not in VALID_STATUSES:
                raise ValueError(
                    f"invalid status: {clean_status!r}; expected one of {VALID_STATUSES}"
                )
            status = clean_status
        updated = TaskRecord(
            id=task.id,
            title=new_title,
            status=task.status if status is None else status,
            payload=task.payload if payload is None else dict(payload),
            created_at=task.created_at,
            updated_at=_timestamp(),
            version=task.version + 1,
            claimed_by=task.claimed_by,
            claimed_at=task.claimed_at,
        )
        self._write(updated)
        return updated

    def claim(self, task_id: int | str, claimant: str) -> TaskRecord | None:
        clean_claimant = claimant.strip()
        if not clean_claimant:
            raise ValueError("claimant is required")
        with self.locked():
            task = self.get(task_id)
            if task.status != PENDING:
                return None
            now = _timestamp()
            claimed = TaskRecord(
                id=task.id,
                title=task.title,
                status=CLAIMED,
                payload=task.payload,
                created_at=task.created_at,
                updated_at=now,
                version=task.version + 1,
                claimed_by=clean_claimant,
                claimed_at=now,
            )
            self._write(claimed)
            return claimed

    def _next_id(self) -> int:
        current = 0
        if self.highwatermark_path.exists():
            text = self.highwatermark_path.read_text(encoding="utf-8").strip()
            current = int(text or "0")
        next_id = current + 1
        _atomic_write(self.highwatermark_path, f"{next_id}\n")
        return next_id

    def _write(self, task: TaskRecord) -> None:
        _atomic_write(
            self._task_path(task.id),
            json.dumps(asdict(task), ensure_ascii=False, indent=2) + "\n",
        )

    def _task_path(self, task_id: int | str) -> Path:
        return self.tasks_dir / f"{int(task_id)}.json"

    @contextmanager
    def locked(self) -> Iterator[None]:
        import filelock

        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = self.tasks_dir / ".lock_file"
        lock = filelock.FileLock(lock_file, timeout=self.lock_timeout_seconds)
        try:
            with lock:
                yield
        except filelock.Timeout as exc:
            raise TimeoutError(
                f"timed out waiting for task store lock: {lock_file}"
            ) from exc


def _decode(text: str) -> TaskRecord:
    data = json.loads(text)
    return TaskRecord(
        id=int(data["id"]),
        title=str(data["title"]),
        status=str(data["status"]),
        payload=dict(data.get("payload") or {}),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
        version=int(data.get("version", 1)),
        claimed_by=data.get("claimed_by"),
        claimed_at=data.get("claimed_at"),
    )


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def resolve_task_dependencies(tasks: list[TaskRecord]) -> list[TaskRecord]:
    """根据 blocked_by 依赖关系对任务进行拓扑排序，返回可直接执行的任务顺序。"""
    task_map = {t.id: t for t in tasks}
    visited: dict[int, int] = {}  # 0: visiting, 1: visited
    result = []

    def visit(task_id: int):
        if task_id in visited:
            if visited[task_id] == 0:
                raise ValueError(
                    f"Circular dependency detected containing task #{task_id}"
                )
            return

        visited[task_id] = 0
        task = task_map.get(task_id)
        if task:
            for dep_id in _parse_blocked_by(task):
                if dep_id in task_map:
                    visit(dep_id)

        visited[task_id] = 1
        if task:
            result.append(task)

    for task in tasks:
        if task.id not in visited:
            visit(task.id)

    return result


def resolve_blocked(tasks: list[TaskRecord]) -> list[dict[str, Any]]:
    """找出所有被阻塞的任务及其阻塞依赖。"""
    task_map = {t.id: t for t in tasks}
    completed_ids = {t.id for t in tasks if t.status == COMPLETED}
    blocked: list[dict[str, Any]] = []
    for t in tasks:
        if t.status == COMPLETED:
            continue
        blocked_by = _parse_blocked_by(t)
        blocking = [dep for dep in blocked_by if dep not in completed_ids]
        if blocking:
            blocking_names = [
                task_map[d].title if d in task_map else f"#{d}" for d in blocking
            ]
            blocked.append(
                {
                    "task_id": t.id,
                    "task_title": t.title,
                    "blocked_by_ids": blocking,
                    "blocked_by_titles": blocking_names,
                }
            )
    return blocked


def advance_task(
    store: TaskStore, task_id: int | str, expected_version: int | None = None
) -> list[TaskRecord]:
    """完成任务并自动解除下游依赖阻塞。

    将 task_id 标记为 completed，然后查找所有 blocked_by 包含此任务的
    待办任务，将它们的 block 状态移除（如果已无其他阻塞则解除阻塞）。
    返回所有受影响的任务列表。
    """
    with store.locked():
        updated = store._apply_update(
            task_id, status=COMPLETED, expected_version=expected_version
        )
        affected = [updated]

        all_tasks = store.list()
        for t in all_tasks:
            if t.status == COMPLETED or t.id == int(task_id):
                continue
            blocked_by = _parse_blocked_by(t)
            if int(task_id) in blocked_by:
                remaining = [d for d in blocked_by if d != int(task_id)]
                new_payload = dict(t.payload)
                if remaining:
                    new_payload["blocked_by"] = remaining
                else:
                    new_payload.pop("blocked_by", None)
                store._apply_update(t.id, payload=new_payload)
                affected.append(store.get(t.id))

    return affected


def _parse_blocked_by(task: TaskRecord) -> list[int]:
    """从 TaskRecord payload 中提取 blocked_by 依赖列表。"""
    blocked_by = task.payload.get("blocked_by")
    if isinstance(blocked_by, list):
        return [item for item in blocked_by if type(item) is int]
    return []


def render_kanban_view(tasks: list[TaskRecord]) -> str:
    """输出美化的终端看板视图。"""
    categories: dict[str, list[TaskRecord]] = {
        PENDING: [],
        CLAIMED: [],
        COMPLETED: [],
        "[unknown]": [],
    }
    for t in tasks:
        cat = t.status if t.status in (PENDING, CLAIMED, COMPLETED) else "[unknown]"
        categories[cat].append(t)

    lines = ["=== TASK KANBAN VIEW ==="]
    for status, list_tasks in categories.items():
        if status == "[unknown]" and not list_tasks:
            continue
        lines.append(f"\n[{status.upper()}] ({len(list_tasks)})")
        if not list_tasks:
            lines.append("  (No tasks)")
            continue
        for t in list_tasks:
            fl_info = ""
            fl = t.payload.get("feature_list")
            if isinstance(fl, list) and fl:
                completed = sum(
                    1
                    for item in fl
                    if isinstance(item, dict) and item.get("status") == COMPLETED
                )
                fl_info = f" ({completed}/{len(fl)} subtasks)"
            blocked_by = t.payload.get("blocked_by")
            block_info = f" [Blocked by: {blocked_by}]" if blocked_by else ""
            lines.append(f"  - #{t.id}: {t.title}{fl_info}{block_info}")
    if categories["[unknown]"]:
        lines.append(
            f"\n[WARNING] {len(categories['[unknown]'])} task(s) have unrecognized status."
        )
    return "\n".join(lines)


def _create_task(store: TaskStore, args: ToolInput) -> str:
    title = str(args.get("title", "")).strip()
    if not title:
        raise ValueError("title is required")
    blocked_by = args.get("blocked_by")
    payload: dict[str, Any] = {}
    if blocked_by:
        payload["blocked_by"] = blocked_by
    task = store.create(title, payload)
    return f"Created task #{task.id}: '{task.title}' (status: {task.status})"


def _update_task(store: TaskStore, args: ToolInput) -> str:
    task_id = args.get("id")
    if task_id is None:
        raise ValueError("id is required")
    title = args.get("title")
    status = args.get("status")
    expected_version = args.get("expected_version")

    current = store.get(task_id)
    payload = dict(current.payload)
    if args.get("blocked_by"):
        payload["blocked_by"] = args.get("blocked_by")

    try:
        task = store.update(
            task_id,
            title=title,
            status=status,
            payload=payload,
            expected_version=int(expected_version)
            if expected_version is not None
            else None,
        )
    except ConcurrentModificationError:
        latest = store.get(task_id)
        return (
            f"Concurrent modification: task #{latest.id} current version is "
            f"{latest.version}, you expected {expected_version}. "
            f"Please re-read and retry."
        )
    return f"Updated task #{task.id}: status={task.status}, version={task.version}"


def _claim_task(store: TaskStore, args: ToolInput) -> str:
    task_id = args.get("task_id")
    claimant = str(args.get("claimant", "")).strip()
    if task_id is None:
        raise ValueError("task_id is required")
    if not claimant:
        raise ValueError("claimant is required")
    claimed = store.claim(task_id, claimant)
    if claimed is None:
        return f"Task #{task_id} is not pending (already claimed or completed)"
    return f"Claimed task #{claimed.id} for {claimant} (version={claimed.version})"


def _list_tasks(store: TaskStore, args: ToolInput) -> str:
    view = str(args.get("view", "kanban")).strip().lower()
    tasks = store.list()
    if not tasks:
        return "No tasks in the store."
    if view == "kanban":
        return render_kanban_view(tasks)
    if view == "topological":
        try:
            sorted_tasks = resolve_task_dependencies(tasks)
            lines = ["=== TOPOLOGICAL TASK LIST ==="]
            for t in sorted_tasks:
                blocked_by = t.payload.get("blocked_by")
                block_info = f" [Blocked by: {blocked_by}]" if blocked_by else ""
                lines.append(f"  - #{t.id} ({t.status}): {t.title}{block_info}")
            return "\n".join(lines)
        except ValueError as e:
            return f"Dependency Resolution Error: {e}"
    lines = ["=== TASK LIST ==="]
    for t in tasks:
        lines.append(f"  - #{t.id} [{t.status}]: {t.title}")
    return "\n".join(lines)


ADVANCE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer", "description": "ID of the task to mark as completed"},
        "expected_version": {
            "type": "integer",
            "description": "Optimistic lock: fail if current version does not match",
        },
    },
    "required": ["id"],
    "additionalProperties": False,
}

RESOLVE_BLOCKED_SCHEMA = {
    "type": "object",
    "properties": {
        "view": {
            "type": "string",
            "enum": ["summary", "detail"],
            "description": "Summary or detailed view of blocked tasks",
        }
    },
    "additionalProperties": False,
}


def _advance_task(store: TaskStore, args: ToolInput) -> str:
    task_id = args.get("id")
    if task_id is None:
        raise ValueError("id is required")
    expected_version = args.get("expected_version")
    try:
        affected = advance_task(
            store,
            task_id,
            expected_version=int(expected_version)
            if expected_version is not None
            else None,
        )
    except ConcurrentModificationError:
        latest = store.get(task_id)
        return (
            f"Concurrent modification: task #{latest.id} current version is "
            f"{latest.version}, you expected {expected_version}. "
            f"Please re-read and retry."
        )
    names = [f"#{t.id} ({t.status}): {t.title}" for t in affected]
    return f"Advanced task #{task_id}.\nAffected:\n" + "\n".join(names)


def _resolve_blocked(store: TaskStore, args: ToolInput) -> str:
    tasks = store.list()
    blocked = resolve_blocked(tasks)
    if not blocked:
        return "No blocked tasks. All dependencies are satisfied."
    lines = ["=== BLOCKED TASKS ==="]
    for b in blocked:
        blockers = ", ".join(b["blocked_by_titles"])
        lines.append(f"  - #{b['task_id']} {b['task_title']} [Blocked by: {blockers}]")
    return "\n".join(lines)


def _get_task(store: TaskStore, args: ToolInput) -> str:
    task_id = args.get("id")
    if task_id is None:
        raise ValueError("id is required")
    try:
        task = store.get(task_id)
        return json.dumps(asdict(task), ensure_ascii=False, indent=2)
    except KeyError:
        return f"Error: Task #{task_id} not found."


def build_task_tools(store: TaskStore) -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="create_task",
            description="Create a durable task graph node. Expose title and blocked_by dependencies.",
            input_hint='{"title": "Implement X", "blocked_by": [1]}',
            handler=partial(_create_task, store),
            schema=CREATE_TASK_SCHEMA,
            group="tasks",
        ),
        ToolSpec(
            name="update_task",
            description="Update task attributes, status (pending/claimed/completed), title, or blocked_by. Pass expected_version for optimistic locking.",
            input_hint='{"id": 1, "status": "completed", "expected_version": 2}',
            handler=partial(_update_task, store),
            schema=UPDATE_TASK_SCHEMA,
            group="tasks",
        ),
        ToolSpec(
            name="claim_task",
            description="Atomically claim a pending task for an agent. Returns failure message if task is already claimed or completed.",
            input_hint='{"task_id": 1, "claimant": "agent_a"}',
            handler=partial(_claim_task, store),
            schema=CLAIM_TASK_SCHEMA,
            group="tasks",
        ),
        ToolSpec(
            name="advance_task",
            description="Mark a task as completed and auto-unblock its dependents. Use instead of update_task for completing tasks with blocked_by dependencies.",
            input_hint='{"id": 1}',
            handler=partial(_advance_task, store),
            schema=ADVANCE_TASK_SCHEMA,
            group="tasks",
        ),
        ToolSpec(
            name="list_tasks",
            description="List task graph nodes. Supports 'kanban', 'topological', or 'raw' views.",
            input_hint='{"view": "kanban"}',
            handler=partial(_list_tasks, store),
            schema=LIST_TASKS_SCHEMA,
            read_only=True,
            group="tasks",
        ),
        ToolSpec(
            name="get_task",
            description="Retrieve detailed fields and full payload of a single task by its integer ID.",
            input_hint='{"id": 1}',
            handler=partial(_get_task, store),
            schema=GET_TASK_SCHEMA,
            read_only=True,
            group="tasks",
        ),
        ToolSpec(
            name="resolve_blocked",
            description="Show which tasks are blocked by unfinished dependencies.",
            input_hint="{}",
            handler=partial(_resolve_blocked, store),
            schema=RESOLVE_BLOCKED_SCHEMA,
            read_only=True,
            group="tasks",
        ),
    )
