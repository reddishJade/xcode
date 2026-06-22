"""Memory 工具注册。

该模块仅暴露只读检索工具；记忆写入由压缩 consolidation 或显式
`/memory add` 命令完成。
"""

from __future__ import annotations

from typing import cast

from xcode.harness.skills import ToolInput, ToolSpec

from .manager import MemoryLayerFilter, MemoryManager


def build_memory_tools(manager: MemoryManager) -> tuple[ToolSpec, ...]:
    """构建 opt-in memory 工具组。"""

    def search_memory(data: ToolInput) -> str:
        """跨项目级与用户级记忆检索并渲染来源。"""
        query = str(data.get("query", "")).strip()
        if not query:
            return "query is required"
        limit = _parse_limit(data.get("limit", 3))
        scope = _optional_text(data.get("scope"))
        layer = str(data.get("layer", "all"))
        if layer not in {"all", "project", "user"}:
            return "layer must be one of: all, project, user"

        records = manager.search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=cast(MemoryLayerFilter, layer),
        )
        if not records:
            return f"No memory matching {query!r}."

        rendered = [
            f"[{record.layer}] score={record.score:.3f}\n{record.block.strip()}"
            for record in records
        ]
        return "\n\n".join(rendered)

    return (
        ToolSpec(
            name="search_memory",
            description=(
                "Search project and user memory for prior solutions, constraints, "
                "files, and takeaways relevant to the current task."
            ),
            input_hint=(
                'JSON: {"query": "provider timeout", "limit": 3, '
                '"scope": "providers", "layer": "all"}'
            ),
            handler=search_memory,
            schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language memory search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional scope used to rerank matching records.",
                    },
                    "layer": {
                        "type": "string",
                        "enum": ["all", "project", "user"],
                        "default": "all",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            read_only=True,
            group="memory",
            prompt_snippet=(
                "Search opt-in project and user memory when prior decisions or "
                "solutions may affect the task."
            ),
        ),
    )


def _parse_limit(value: object) -> int:
    """将工具输入限制到安全结果范围。"""
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return 3
    else:
        return 3
    return min(max(parsed, 1), 10)


def _optional_text(value: object) -> str | None:
    """将可选输入规范化为非空文本。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
