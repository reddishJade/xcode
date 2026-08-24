"""会话级 todo 状态。"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from typing import Literal

type TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
type TodoPriority = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class TodoItem:
    """单个会话 todo。"""

    id: str
    content: str
    status: TodoStatus
    priority: TodoPriority | None = None


class SessionTodoState:
    """线程安全的会话 todo 真值源。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: tuple[TodoItem, ...] = ()

    def replace(self, raw_items: object) -> tuple[TodoItem, ...]:
        """校验并完整替换 todo 清单。"""
        items = _parse_items(raw_items)
        with self._lock:
            self._items = items
        return items

    def snapshot(self) -> tuple[TodoItem, ...]:
        """返回当前不可变快照。"""
        with self._lock:
            return self._items

    def to_dicts(self) -> list[dict[str, str | None]]:
        """返回 JSON 可序列化状态。"""
        return [asdict(item) for item in self.snapshot()]

    def render_context(self) -> str:
        """渲染给后续 turn/压缩后恢复用的上下文。"""
        items = self.snapshot()
        if not items:
            return ""
        lines = ["<session-todo>"]
        lines.extend(_render_item(item) for item in items)
        lines.append("</session-todo>")
        return "\n".join(lines)


def _parse_items(raw_items: object) -> tuple[TodoItem, ...]:
    if not isinstance(raw_items, list):
        raise TypeError("todos must be an array")
    items: list[TodoItem] = []
    seen_ids: set[str] = set()
    in_progress_count = 0
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            raise TypeError("todo items must be objects")
        content = str(raw_item.get("content", "")).strip()
        status = raw_item.get("status", "pending")
        priority = raw_item.get("priority")
        item_id = str(raw_item.get("id", "")).strip() or _default_id(content, index)
        if not item_id:
            raise ValueError("todo id must not be empty")
        if item_id in seen_ids:
            raise ValueError(f"duplicate todo id: {item_id}")
        if not content:
            raise ValueError(f"todo content must not be empty: {item_id}")
        if status not in {"pending", "in_progress", "completed", "cancelled"}:
            raise ValueError(f"invalid todo status for {item_id}: {status}")
        if priority is not None and priority not in {"high", "medium", "low"}:
            raise ValueError(f"invalid todo priority for {item_id}: {priority}")
        if status == "in_progress":
            in_progress_count += 1
        seen_ids.add(item_id)
        items.append(TodoItem(item_id, content, status, priority))
    if in_progress_count > 1:
        raise ValueError("at most one todo item may be in_progress")
    return tuple(items)


def _default_id(content: str, index: int) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", content.lower()).strip("-")
    return f"todo-{stem or index}"


def _render_item(item: TodoItem) -> str:
    priority = f' priority="{item.priority}"' if item.priority is not None else ""
    return f'- id="{item.id}" status="{item.status}"{priority}: {item.content}'
