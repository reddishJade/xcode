from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from xcode.agent.types import ToolSpec as CoreToolSpec


_todos: list[dict[str, Any]] = []
_lock = threading.Lock()


def _reset() -> None:
    with _lock:
        _todos.clear()


def build_todowrite_tool() -> CoreToolSpec:
    def handler(data: dict[str, Any], _on_update: Callable[[str], None] | None = None) -> str:
        raw = data.get("todos")
        if not isinstance(raw, list):
            return 'Error: "todos" must be an array'
        validated: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                return "Error: each todo must be an object"
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).strip()
            if not content:
                return "Error: todo content must not be empty"
            if status not in {"pending", "in_progress", "completed", "cancelled"}:
                return f"Error: invalid status: {status}"
            entry: dict[str, Any] = {"content": content, "status": status}
            priority = item.get("priority")
            if priority is not None:
                if priority not in {"high", "medium", "low"}:
                    return f"Error: invalid priority: {priority}"
                entry["priority"] = priority
            validated.append(entry)
        with _lock:
            _todos.clear()
            _todos.extend(validated)
        return json.dumps({"todos": validated}, ensure_ascii=False, indent=2)

    return CoreToolSpec(
        name="todowrite",
        description=(
            "Create and maintain a structured task list for the current session. "
            "Use it to track progress during multi-step work. "
            "Provide the full list each time to replace the previous state."
        ),
        input_hint='JSON: {"todos": [{"content": "Implement X", "status": "in_progress", "priority": "high"}]}',
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Description of the task"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                                "description": "Current status",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                    "description": "Full todo list (replaces previous state)",
                }
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
        prompt_snippet="Track tasks and progress",
        prompt_guidelines=(
            "Use todowrite to maintain a persistent todo list during multi-step tasks.",
            "Provide the full list each time; the system replaces the previous state.",
            "Mark items completed when done, update priorities as needed.",
        ),
    )
