"""检查系统已安装的外部工具（如 fd、ripgrep）。"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("xcode.coding_agent.tools.tools_manager")

ToolResolver = Callable[[str], str | None]


@dataclass(frozen=True)
class ExternalToolDefinition:
    display_name: str
    candidate_names: tuple[str, ...]


_TOOLS: dict[str, ExternalToolDefinition] = {
    "fd": ExternalToolDefinition(display_name="fd", candidate_names=("fd", "fdfind")),
    "rg": ExternalToolDefinition(display_name="rg", candidate_names=("rg",)),
}


def get_tool_path(tool: str) -> str | None:
    """检查工具是否在系统 PATH 中可用。"""
    definition = _TOOLS.get(tool)
    if definition is None:
        return None

    return _resolve_tool_path(definition, shutil.which)


def _resolve_tool_path(
    definition: ExternalToolDefinition,
    resolver: ToolResolver,
) -> str | None:
    for name in definition.candidate_names:
        found = resolver(name)
        if found:
            return found
    return None


def ensure_tool(tool: str, silent: bool = False) -> str | None:
    """确保工具可用。

    检查工具是否在系统 PATH 中，返回工具路径，若不可用则返回 None。
    """
    existing = get_tool_path(tool)
    if existing:
        return existing

    definition = _TOOLS.get(tool)
    if definition is None:
        return None

    if not silent:
        logger.warning(
            "%s not found in PATH. Please install it manually.",
            definition.display_name,
        )

    return None
