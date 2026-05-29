from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from ..harness.skills import ToolSpec


class PluginManager:
    """动态扫描与加载本地插件目录 (.local/plugins/) 的管理器。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.plugins_dir = project_root / ".local" / "plugins"
        self.loaded_modules: dict[str, Any] = {}

    def scan_and_load(self) -> dict[str, Any]:
        """扫描插件目录并动态加载 Python 模块。"""
        if not self.plugins_dir.exists():
            return {"tools": [], "hooks": {}, "skills": []}

        tools: list[ToolSpec] = []
        hooks: dict[str, list[Callable]] = {}
        skills: list[Any] = []

        # 遍历 .py 文件（排除 __init__.py）
        for path in sorted(self.plugins_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            module_name = f"xcode_plugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    # 避免污染 sys.modules
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    self.loaded_modules[path.stem] = module

                    # 1. 收集暴露的自定义工具
                    if hasattr(module, "exposed_tools"):
                        mod_tools = getattr(module, "exposed_tools")
                        if isinstance(mod_tools, list):
                            for t in mod_tools:
                                if isinstance(t, ToolSpec):
                                    tools.append(t)

                    # 2. 收集暴露的生命周期 Hooks
                    if hasattr(module, "exposed_hooks"):
                        mod_hooks = getattr(module, "exposed_hooks")
                        if isinstance(mod_hooks, dict):
                            for event, cb in mod_hooks.items():
                                if callable(cb):
                                    hooks.setdefault(event, []).append(cb)

                    # 3. 收集暴露的技能 SOPs
                    if hasattr(module, "exposed_skills"):
                        mod_skills = getattr(module, "exposed_skills")
                        if isinstance(mod_skills, list):
                            skills.extend(mod_skills)
            except Exception as e:
                print(f"Warning: Failed to dynamically load plugin {path.name}: {e}")

        return {
            "tools": tools,
            "hooks": hooks,
            "skills": skills,
        }
