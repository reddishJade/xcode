"""Memory 工具注册。

该模块仅暴露只读检索工具；记忆写入由压缩 consolidation 或显式
`/memory add` 命令完成。
"""

from __future__ import annotations

from typing import cast

from xcode.harness.skills import ToolInput, ToolSpec

from .manager import MemoryLayerFilter, MemoryManager, MemoryRetrievalContext


def build_memory_tools(manager: MemoryManager) -> tuple[ToolSpec, ...]:
    """构建 opt-in memory 工具组。"""

    def search_memory(data: ToolInput) -> str:
        """跨项目级与用户级记忆检索并渲染来源。"""
        query = str(data.get("query", "")).strip()
        if not query:
            return "query is required"
        limit = _parse_limit(data.get("limit", 3))
        scope = _optional_text(data.get("scope"))
        current_file = _optional_text(data.get("current_file"))
        task_phase = _optional_text(data.get("task_phase"))
        layer = str(data.get("layer", "all"))
        if layer not in {"all", "project", "user"}:
            return "layer must be one of: all, project, user"
        symbols = _optional_list(data.get("symbols"))
        error_messages = _optional_list(data.get("error_messages"))
        modules = _optional_list(data.get("modules"))
        recent_files = _optional_list(data.get("recent_files"))

        records = manager.search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=cast(MemoryLayerFilter, layer),
            source="tool",
            retrieval_context=MemoryRetrievalContext(
                query=query,
                scope=scope,
                current_file=current_file,
                symbols=symbols,
                error_messages=error_messages,
                task_phase=task_phase,
                modules=modules,
                recent_files=recent_files,
            ),
        )
        if not records:
            return f"No memory matching {query!r}."

        rendered = [manager.render_search_result(record) for record in records]
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
                '"scope": "providers", "current_file": "src/provider.py", '
                '"symbols": ["ProviderClient"], "error_messages": ["connection timeout"], '
                '"task_phase": "debug", "modules": ["providers"], '
                '"recent_files": ["src/provider.py"], "layer": "all"}'
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
                    "current_file": {
                        "type": "string",
                        "description": "Current file relevant to the task.",
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant code symbols for retrieval and reranking.",
                    },
                    "error_messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant error messages or failure signatures.",
                    },
                    "task_phase": {
                        "type": "string",
                        "description": "Current task phase, such as debug, implement, or verify.",
                    },
                    "modules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant project modules or subsystems.",
                    },
                    "recent_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recent files already made relevant by the task.",
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


def _optional_list(value: object) -> tuple[str, ...]:
    """将工具输入规范化为文本元组。"""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value]
        return tuple(item for item in normalized if item)
    return ()
