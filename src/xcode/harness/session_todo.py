"""会话级轻量待办状态与工具。"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Literal

type TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
type TodoPriority = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class TodoItem:
    """单个会话待办项。"""

    id: str
    content: str
    status: TodoStatus
    priority: TodoPriority | None = None


class SessionTodoState:
    """线程安全的当前会话待办真值源。"""

    def __init__(self) -> None:
        """初始化空待办清单。"""
        self._lock = threading.Lock()
        self._items: tuple[TodoItem, ...] = ()

    def replace(self, raw_items: object) -> tuple[TodoItem, ...]:
        """校验并完整替换当前清单。"""
        items = _parse_items(raw_items)
        with self._lock:
            self._items = items
        return items

    def snapshot(self) -> tuple[TodoItem, ...]:
        """返回不可变清单快照。"""
        with self._lock:
            return self._items

    def to_dicts(self) -> list[dict[str, str | None]]:
        """返回 JSON 可序列化清单。"""
        return [asdict(item) for item in self.snapshot()]

    def render_context(self) -> str:
        """渲染压缩后仍会重新注入的会话待办上下文。"""
        items = self.snapshot()
        if not items:
            return ""
        lines = ["<session-todo>"]
        lines.extend(_render_item(item) for item in items)
        lines.append("</session-todo>")
        return "\n".join(lines)


def _parse_items(raw_items: object) -> tuple[TodoItem, ...]:
    """解析并校验完整替换载荷。"""
    if not isinstance(raw_items, list):
        raise ValueError("todos must be an array")
    items: list[TodoItem] = []
    seen_ids: set[str] = set()
    in_progress_count = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("todo items must be objects")
        item_id = str(raw_item.get("id", "")).strip()
        content = str(raw_item.get("content", "")).strip()
        status = raw_item.get("status")
        priority = raw_item.get("priority")
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


def _render_item(item: TodoItem) -> str:
    priority = f' priority="{item.priority}"' if item.priority is not None else ""
    return f'- id="{item.id}" status="{item.status}"{priority}: {item.content}'
