"""会话待办工具。"""

from __future__ import annotations

import json
from dataclasses import asdict

from xcode.harness.session_todo import SessionTodoState
from xcode.harness.skills import ToolInput, ToolSpec


def build_todo_tools(state: SessionTodoState) -> tuple[ToolSpec, ...]:
    """构建默认可用的轻量待办工具。"""

    def todowrite(args: ToolInput) -> str:
        items = state.replace(args.get("todos"))
        return json.dumps(
            {"todos": [asdict(item) for item in items]},
            ensure_ascii=False,
        )

    return (
        ToolSpec(
            name="todowrite",
            description=(
                "Replace the current xcode session todo list. Use stable ids so "
                "RunState and transcript resume can preserve items, and keep at "
                "most one item in progress."
            ),
            input_hint=(
                '{"todos":[{"id":"design","content":"Design interface",'
                '"status":"in_progress","priority":"high"}]}'
            ),
            handler=todowrite,
            schema={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "content": {"type": "string", "minLength": 1},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                        "cancelled",
                                    ],
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                            },
                            "required": ["id", "content", "status"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["todos"],
                "additionalProperties": False,
            },
            group="core",
        ),
    )
