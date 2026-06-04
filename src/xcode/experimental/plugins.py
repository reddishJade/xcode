from __future__ import annotations

import ast
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ..harness.skills import ToolSpec


class PluginManifest:
    """插件清单，声明插件的元数据和暴露的能力。"""

    __slots__ = ("name", "version", "tools", "hooks", "skills")

    def __init__(
        self,
        name: str,
        version: str = "0.1.0",
        tools: bool = False,
        hooks: bool = False,
        skills: bool = False,
    ) -> None:
        self.name = name
        self.version = version
        self.tools = tools
        self.hooks = hooks
        self.skills = skills


def _validate_exposed_tools(tools: object) -> list[ToolSpec]:
    if not isinstance(tools, list):
        return []
    valid: list[ToolSpec] = []
    for item in tools:
        if isinstance(item, ToolSpec):
            if not item.name or not callable(item.handler):
                continue
            valid.append(item)
    return valid


def _validate_exposed_hooks(hooks: object) -> dict[str, list[Callable]]:
    if not isinstance(hooks, dict):
        return {}
    valid: dict[str, list[Callable]] = {}
    for event, cb in hooks.items():
        if not isinstance(event, str):
            continue
        if callable(cb):
            valid.setdefault(event, []).append(cb)
    return valid


def _validate_exposed_skills(skills: object) -> list[Any]:
    if not isinstance(skills, list):
        return []
    return list(skills)


def _build_plugin_globals(manifest: PluginManifest) -> dict[str, Any]:
    """构造插件执行用的全局命名空间。

    这是 in-process exec 的便利设施，不是安全边界。
    插件代码与宿主共享进程空间，可访问文件系统和网络。
    """
    restricted = {
        "__builtins__": __builtins__,
        "__name__": manifest.name,
        "__doc__": None,
        "ToolSpec": ToolSpec,
        "Any": Any,
        "Callable": Callable,
        "exposed_tools": [],
        "exposed_hooks": {},
        "exposed_skills": [],
    }
    return restricted


class PluginManager:
    """受限插件加载器。

    从 .local/plugins/*.py 加载 Python 文件，按约定收集
    exposed_tools / exposed_hooks / exposed_skills。

    注意：
    - 这是 in-process exec 加载，不是安全沙箱。
    - 插件代码可访问宿主全部文件系统和网络。
    - 仅在用户显式启用 plugins 或 experimental group 时加载。
    - 建议仅在受信任环境中使用。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.plugins_dir = project_root / ".local" / "plugins"
        self.loaded_modules: dict[str, Any] = {}

    def scan_and_load(self) -> dict[str, Any]:
        """扫描插件目录并加载 Python 模块。"""
        if not self.plugins_dir.exists():
            return {"tools": [], "hooks": {}, "skills": []}

        tools: list[ToolSpec] = []
        hooks: dict[str, list[Callable]] = {}
        skills: list[Any] = []

        for path in sorted(self.plugins_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            result = self._load_plugin(path)
            if result is None:
                continue
            tools.extend(result.get("tools", []))
            for event, cbs in result.get("hooks", {}).items():
                hooks.setdefault(event, []).extend(cbs)
            skills.extend(result.get("skills", []))

        return {
            "tools": tools,
            "hooks": hooks,
            "skills": skills,
        }

    def _load_plugin(self, path: Path) -> dict[str, Any] | None:
        module_name = f"xcode_plugin_{path.stem}"
        try:
            manifest = self._extract_manifest(path)
            plugin_globals = _build_plugin_globals(
                manifest or PluginManifest(name=path.stem)
            )

            source = path.read_text(encoding="utf-8")
            exec(compile(source, path.name, "exec"), plugin_globals)

            exposed_tools = _validate_exposed_tools(
                plugin_globals.get("exposed_tools", [])
            )
            exposed_hooks = _validate_exposed_hooks(
                plugin_globals.get("exposed_hooks", {})
            )
            exposed_skills = _validate_exposed_skills(
                plugin_globals.get("exposed_skills", [])
            )

            # 清单校验：声明了能力但未导出时拒绝（仅限有 manifest 的插件）
            if manifest and manifest.tools and not exposed_tools:
                return None
            if manifest and manifest.hooks and not exposed_hooks:
                return None
            if manifest and manifest.skills and not exposed_skills:
                return None

            module = type(sys)(module_name)
            setattr(module, "exposed_tools", exposed_tools)
            setattr(module, "exposed_hooks", exposed_hooks)
            setattr(module, "exposed_skills", exposed_skills)
            if manifest:
                setattr(module, "__manifest__", manifest)
            self.loaded_modules[path.stem] = module

            return {
                "tools": exposed_tools,
                "hooks": exposed_hooks,
                "skills": exposed_skills,
            }

        except Exception as e:
            print(f"Warning: Failed to load plugin {path.name}: {e}")
            return None

    def _extract_manifest(self, path: Path) -> PluginManifest | None:
        """用 AST 解析提取 __plugin_name__ 等模块级常量。"""
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            return None

        try:
            tree = ast.parse(source, filename=path.name)
        except SyntaxError:
            return None

        manifest_name = None
        manifest_version = None
        manifest_tools = False
        manifest_hooks = False
        manifest_skills = False

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            value = node.value

            if name == "__plugin_name__":
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    manifest_name = value.value
            elif name == "__plugin_version__":
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    manifest_version = value.value
            elif name in ("__plugin_tools__", "__plugin_hooks__", "__plugin_skills__"):
                if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                    if name == "__plugin_tools__":
                        manifest_tools = value.value
                    elif name == "__plugin_hooks__":
                        manifest_hooks = value.value
                    elif name == "__plugin_skills__":
                        manifest_skills = value.value

        if not manifest_name:
            return None

        return PluginManifest(
            name=manifest_name,
            version=manifest_version or "0.1.0",
            tools=manifest_tools,
            hooks=manifest_hooks,
            skills=manifest_skills,
        )
