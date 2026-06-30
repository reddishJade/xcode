"""CLI 工具目录与生产 registry builder 一致性测试。"""

from __future__ import annotations

import ast
from pathlib import Path

from xcode.cli.tool_catalog import (
    build_tool_catalog,
    CATALOG_COVERED_BUILDERS,
)
import pytest


class ToolCatalogConsistencyTests:
    """防止新增生产工具 builder 后遗漏 CLI catalog。"""

    def test_catalog_covers_production_tool_builders(self) -> None:
        """assembly 和 product registry 中的 builder 必须显式登记。"""
        root = Path(__file__).resolve().parents[1]
        production_builders: set[str] = set()
        for relative_path in (
            "coding_agent/registry.py",
            "harness/assembly.py",
        ):
            source = (root / relative_path).read_text(encoding="utf-8")
            tree = ast.parse(source)
            production_builders.update(_tool_builder_calls(tree))

        assert production_builders - CATALOG_COVERED_BUILDERS == set()

    def test_catalog_contains_runtime_composed_tools(self) -> None:
        """assembly 直接组合的 session、search、skill、subagent 工具可见。"""
        catalog = build_tool_catalog()

        assert "update_todo" in catalog["session"]
        assert "search_tools" in catalog["core"]
        assert "apply_patch" in catalog["core"]
        assert "create_worktree_task" in catalog["worktree"]
        assert "load_skill" in catalog["skills"]
        assert catalog["subagent"] == {"subagent"}


def _tool_builder_calls(tree: ast.AST) -> set[str]:
    """收集生产装配源码中的工具 builder 调用。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name is None:
            continue
        if name == "build_bash_tool" or name.endswith("_tools"):
            names.add(name)
        elif name in {"build_load_skill_tool", "build_search_tools_tool"}:
            names.add(name)
    return names


def _called_name(node: ast.expr) -> str | None:
    """返回直接或属性调用的函数名。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


if __name__ == "__main__":
    pytest.main()
