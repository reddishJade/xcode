"""Agent 编码工具——工作区文件、代码搜索、shell 执行的工具注册与编排。"""

from __future__ import annotations

from .registry import build_project_scoped_registry

__all__ = [
    "build_project_scoped_registry",
]
