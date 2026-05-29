"""统一工具目录：从工具构建函数自动提取名称和分组。

每次调用 `build_tool_catalog()` 都会在临时上下文中扫描所有注册的构建函数，
返回 `{group: set_of_tool_names}`。新增工具或修改 group 后无需手动更新任何列表。

导入此模块本身没有副作用；build_tool_catalog() 调用各构建函数时使用
自动清理的临时目录作为 project_root，构造阶段不会执行网络请求。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import tempfile
from typing import Any

from xcode.harness.skills import ToolSpec
from xcode.harness.tools import (
    build_bash_tool,
    build_code_tools,
    build_file_tools,
)

from xcode.experimental.worktree import WorktreeTaskRunner, build_worktree_tools
from xcode.experimental.tasks import TaskStore, build_task_tools


def _builders(base_tmp: Path) -> list[tuple[str, Callable[[], Any]]]:
    from xcode.harness.skill_loader import SkillLoader, build_skill_loader_tool

    return [
        ("core", lambda: build_file_tools(base_tmp)),
        ("core", lambda: build_code_tools(base_tmp)),
        ("core", lambda: (build_bash_tool(base_tmp),)),
        (
            "skills",
            lambda: (
                build_skill_loader_tool(
                    SkillLoader(base_tmp / "skills"),
                ),
            ),
        ),
        (
            "worktree",
            lambda: build_worktree_tools(
                WorktreeTaskRunner(base_tmp),
            ),
        ),
        (
            "tasks",
            lambda: build_task_tools(
                TaskStore(base_tmp),
            ),
        ),
    ]


def build_tool_catalog() -> dict[str, set[str]]:
    """扫描所有工具构建函数，返回 {group: set_of_tool_names}。"""
    catalog: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix="xcode-catalog-") as temp_dir:
        for _hint, builder in _builders(Path(temp_dir)):
            specs = builder()
            if not isinstance(specs, (tuple, list)):
                specs = (specs,)
            for spec in specs:
                if not isinstance(spec, ToolSpec):
                    continue
                group = spec.group
                catalog.setdefault(group, set()).add(spec.name)

    if "subagent" not in catalog:
        catalog["subagent"] = {"submit_subagent", "check_subagent", "cancel_subagent"}

    return catalog
