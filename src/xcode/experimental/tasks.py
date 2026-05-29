from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator
import json

from ..harness.skills import ToolSpec
import os
import time
import uuid

"""轻量任务存储。

每个任务保存为 `.local/tasks.json.d/{id}.json`，`.highwatermark` 记录已经分配
过的最大顺序 ID。目录锁用于保护 ID 分配、更新和领取，避免并发写入互相覆盖。
"""


PENDING = "pending"
CLAIMED = "claimed"


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
        with self._locked():
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
    ) -> TaskRecord:
        with self._locked():
            task = self.get(task_id)
            new_title = task.title if title is None else title.strip()
            if not new_title:
                raise ValueError("title is required")
            updated = TaskRecord(
                id=task.id,
                title=new_title,
                status=task.status if status is None else status.strip(),
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
        with self._locked():
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
    def _locked(self) -> Iterator[None]:
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
            blocked_by = task.payload.get("blocked_by")
            dep_ids = []
            if isinstance(blocked_by, int):
                dep_ids.append(blocked_by)
            elif isinstance(blocked_by, list):
                dep_ids.extend([int(x) for x in blocked_by if str(x).isdigit()])
            elif isinstance(blocked_by, str) and blocked_by.isdigit():
                dep_ids.append(int(blocked_by))

            for dep_id in dep_ids:
                if dep_id in task_map:
                    visit(dep_id)

        visited[task_id] = 1
        if task:
            result.append(task)

    for task in tasks:
        if task.id not in visited:
            visit(task.id)

    return result


def render_kanban_view(tasks: list[TaskRecord]) -> str:
    """输出美化的终端看板视图。"""
    categories: dict[str, list[TaskRecord]] = {
        "pending": [],
        "claimed": [],
        "completed": [],
    }
    for t in tasks:
        cat = t.status if t.status in categories else "pending"
        categories[cat].append(t)

    lines = ["=== TASK KANBAN VIEW ==="]
    for status, list_tasks in categories.items():
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
                    if isinstance(item, dict) and item.get("status") == "completed"
                )
                fl_info = f" ({completed}/{len(fl)} subtasks)"
            blocked_by = t.payload.get("blocked_by")
            block_info = f" [Blocked by: {blocked_by}]" if blocked_by else ""
            lines.append(f"  - #{t.id}: {t.title}{fl_info}{block_info}")
    return "\n".join(lines)


def build_task_tools(store: TaskStore) -> tuple[ToolSpec, ...]:
    from ..harness.skills import parse_tool_input

    def create_task(action_input: str) -> str:
        args = parse_tool_input(action_input)
        title = str(args.get("title", "")).strip()
        if not title:
            return "title is required"
        blocked_by = args.get("blocked_by") or args.get("dependencies")
        payload = {}
        if blocked_by:
            payload["blocked_by"] = blocked_by
        if "payload" in args and isinstance(args["payload"], dict):
            payload.update(args["payload"])
        task = store.create(title, payload)
        return f"Created task #{task.id}: '{task.title}' (status: {task.status})"

    def update_task(action_input: str) -> str:
        args = parse_tool_input(action_input)
        task_id = args.get("id")
        if task_id is None:
            return "id is required"
        title = args.get("title")
        status = args.get("status")
        payload_update = args.get("payload")

        current = store.get(task_id)
        payload = dict(current.payload)
        if args.get("blocked_by"):
            payload["blocked_by"] = args.get("blocked_by")
        if isinstance(payload_update, dict):
            payload.update(payload_update)

        task = store.update(task_id, title=title, status=status, payload=payload)
        return f"Updated task #{task.id}: status={task.status}, version={task.version}"

    def list_tasks(action_input: str) -> str:
        args = parse_tool_input(action_input)
        view = str(args.get("view", "kanban")).strip().lower()
        tasks = store.list()
        if not tasks:
            return "No tasks in the store."
        if view == "kanban":
            return render_kanban_view(tasks)
        elif view == "topological":
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
        else:
            lines = ["=== TASK LIST ==="]
            for t in tasks:
                lines.append(f"  - #{t.id} [{t.status}]: {t.title}")
            return "\n".join(lines)

    def get_task(action_input: str) -> str:
        args = parse_tool_input(action_input)
        task_id = args.get("id")
        if task_id is None:
            return "id is required"
        try:
            task = store.get(task_id)
            return json.dumps(asdict(task), ensure_ascii=False, indent=2)
        except KeyError:
            return f"Error: Task #{task_id} not found."

    return (
        ToolSpec(
            name="create_task",
            description="Create a durable task graph node. Expose title, description, and dependencies/blocked_by.",
            input_hint='{"title": "Implement X", "blocked_by": [1]}',
            handler=create_task,
            risk="low",
            schema={
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
                "additionalProperties": True,
            },
            group="tasks",
        ),
        ToolSpec(
            name="update_task",
            description="Update task attributes, status (pending/claimed/completed), title, or dependencies.",
            input_hint='{"id": 1, "status": "completed"}',
            handler=update_task,
            risk="low",
            schema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "ID of the task to update",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status of the task",
                    },
                    "title": {"type": "string", "description": "New title of the task"},
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "IDs of tasks that this task is blocked by",
                    },
                },
                "required": ["id"],
                "additionalProperties": True,
            },
            group="tasks",
        ),
        ToolSpec(
            name="list_tasks",
            description="List task graph nodes. Supports 'kanban', 'topological', or 'raw' views.",
            input_hint='{"view": "kanban"}',
            handler=list_tasks,
            risk="low",
            schema={
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": ["kanban", "topological", "raw"],
                        "description": "View style to render the tasks",
                    }
                },
                "additionalProperties": False,
            },
            read_only=True,
            group="tasks",
        ),
        ToolSpec(
            name="get_task",
            description="Retrieve detailed fields and full payload of a single task by its integer ID.",
            input_hint='{"id": 1}',
            handler=get_task,
            risk="low",
            schema={
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Task ID"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            read_only=True,
            group="tasks",
        ),
    )
