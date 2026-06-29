"""统一工具目录：从工具构建函数自动提取名称和分组。

每次调用 `build_tool_catalog()` 都会在临时上下文中扫描所有注册的构建函数，
返回 `{group: set_of_tool_names}`。新增工具或修改 group 后无需手动更新任何列表。
新增 `build_*_tools()` 入口时必须同步加入 `_builders()`，确保目录和实际 registry
保持一致。

导入此模块本身没有副作用；build_tool_catalog() 调用各构建函数时使用
自动清理的临时目录作为 project_root，构造阶段不会执行网络请求。
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from xcode.harness.skills import ToolSpec

from xcode.coding_agent.tools import (
    build_bash_tool,
    build_file_tools,
    build_glob_tools,
    build_grep_tool,
)
from xcode.experimental.task_store import TaskStore, build_task_tools
from xcode.experimental.worktree import WorktreeTaskRunner, build_worktree_tools
from xcode.harness.session_todo import build_session_todo_tools, SessionTodoState
from xcode.harness.assembly import build_search_tools_tool
from xcode.harness.memory import MemoryManager, build_memory_tools

type ToolCatalogBuilder = Callable[[], tuple[ToolSpec, ...]]

CATALOG_COVERED_BUILDERS = frozenset(
    {
        "build_bash_tool",
        "build_glob_tools",
        "build_grep_tool",
        "build_file_tools",
        "build_load_skill_tool",
        "build_mailbox_tools",
        "build_managed_subagent_tools",
        "build_mcp_tools",
        "build_memory_tools",
        "build_progress_tools",
        "build_search_tools_tool",
        "build_session_todo_tools",
        "build_task_tools",
        "build_worktree_tools",
    }
)


def _builders(base_tmp: Path) -> list[ToolCatalogBuilder]:
    return [
        lambda: build_file_tools(base_tmp),
        lambda: build_glob_tools(base_tmp) + (build_grep_tool(base_tmp),),
        lambda: (build_bash_tool(base_tmp),),
        lambda: build_worktree_tools(
            WorktreeTaskRunner(base_tmp),
        ),
        lambda: build_task_tools(
            TaskStore(base_tmp),
        ),
        lambda: _build_mcp_catalog(base_tmp),
        lambda: _build_mailbox_catalog(base_tmp),
        lambda: _build_progress_catalog(base_tmp),
        lambda: build_memory_tools(MemoryManager(base_tmp)),
        lambda: build_session_todo_tools(SessionTodoState()),
        lambda: (build_search_tools_tool(lambda: ()),),
    ]


def _build_mcp_catalog(base_tmp: Path) -> tuple[ToolSpec, ...]:
    from xcode.harness.mcp import build_mcp_tools

    mcp_config = base_tmp / ".local" / "mcp_config.json"
    if not mcp_config.exists():
        mcp_config.parent.mkdir(parents=True, exist_ok=True)
        mcp_config.write_text("{}", encoding="utf-8")
    return build_mcp_tools(base_tmp)


def _build_mailbox_catalog(base_tmp: Path) -> tuple[ToolSpec, ...]:
    from xcode.experimental.mailbox import AgentMailbox, build_mailbox_tools

    return build_mailbox_tools(AgentMailbox(base_tmp))


def _build_progress_catalog(base_tmp: Path) -> tuple[ToolSpec, ...]:
    from xcode.experimental.orchestration_store import OrchestrationStore
    from xcode.experimental.task_progress import build_progress_tools
    from xcode.experimental.task_store import TaskStore

    return build_progress_tools(TaskStore(base_tmp), OrchestrationStore(base_tmp))


def build_tool_catalog() -> dict[str, set[str]]:
    catalog: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix="xcode-catalog-") as temp_dir:
        for builder in _builders(Path(temp_dir)):
            for spec in builder():
                group = spec.group
                catalog.setdefault(group, set()).add(spec.name)

    if "subagent" not in catalog:
        catalog["subagent"] = {"submit_subagent", "check_subagent", "cancel_subagent"}
    catalog.setdefault("skills", set()).add("load_skill")

    return catalog
