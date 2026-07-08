from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict

from xcode.agent.types import ToolSpec as CoreToolSpec
from xcode.harness.session_todo import SessionTodoState


def build_todowrite_tool(state: SessionTodoState | None = None) -> CoreToolSpec:
    todo_state = state or SessionTodoState()

    def handler(
        data: dict[str, object], _on_update: Callable[[str], None] | None = None
    ) -> str:
        try:
            items = todo_state.replace(data.get("todos"))
        except ValueError as exc:
            return f"Error: {exc}"
        return json.dumps(
            {"todos": [asdict(item) for item in items]},
            ensure_ascii=False,
            indent=2,
        )

    return CoreToolSpec(
        name="todowrite",
        description=(
            "Create and maintain a structured task list for the current session. "
            "Use it to track progress during multi-step work. "
            "Provide the full list each time to replace the previous state."
        ),
        input_hint='JSON: {"todos": [{"id":"implement-x", "content": "Implement X", "status": "in_progress", "priority": "high"}]}',
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Stable id for updating an existing todo",
                            },
                            "content": {
                                "type": "string",
                                "description": "Description of the task",
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "cancelled",
                                ],
                                "description": "Current status",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
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
            "Use stable ids when updating existing items.",
            "Keep at most one item in_progress.",
            "Mark items completed when done, update priorities as needed.",
        ),
    )
