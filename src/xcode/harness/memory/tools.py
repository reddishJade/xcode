"""长期记忆的只读搜索工具。"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from xcode.agent.types import ToolInput, ToolSpec

from .manager import MemoryLayerFilter, MemoryManager


def build_memory_tools(manager: MemoryManager) -> tuple[ToolSpec, ...]:
    """构建一个显式、按需调用的 Memory 搜索工具。"""

    def search_memory(
        data: ToolInput,
        _on_update: Callable[[str], None] | None = None,
    ) -> str:
        query = str(data.get("query", "")).strip()
        if not query:
            return "query is required"
        layer = str(data.get("layer", "all"))
        if layer not in {"all", "project", "user"}:
            return "layer must be one of: all, project, user"
        limit = _parse_limit(data.get("limit", 3))
        scope = str(data.get("scope", "")).strip() or None
        records = manager.search_memory_records(
            query,
            limit=limit,
            layer=cast(MemoryLayerFilter, layer),
            scope=scope,
        )
        if not records:
            return (
                f"No memory matching {query!r}. Try fewer terms or inspect "
                "MEMORY.md directly."
            )
        return "\n\n".join(manager.render_search_result(record) for record in records)

    return (
        ToolSpec(
            name="search_memory",
            description=(
                "Search durable project and user memory for prior rules, "
                "architecture decisions, verified facts, and reusable solutions."
            ),
            input_hint=(
                'JSON: {"query": "provider timeout", "limit": 3, '
                '"scope": "providers", "layer": "all"}'
            ),
            handler=search_memory,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                    },
                    "scope": {"type": "string"},
                    "layer": {
                        "type": "string",
                        "enum": ["all", "project", "user"],
                        "default": "all",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            prompt_snippet=(
                "Use search_memory before asking the user to repeat prior "
                "project decisions or constraints."
            ),
        ),
    )


def _parse_limit(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 3
    return min(max(parsed, 1), 10)
