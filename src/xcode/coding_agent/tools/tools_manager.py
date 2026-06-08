"""检查系统已安装的外部工具（如 fd、ripgrep）。"""

from __future__ import annotations

import logging
import shutil
from typing import Any

logger = logging.getLogger("xcode.coding_agent.tools.tools_manager")

_TOOLS: dict[str, dict[str, Any]] = {
    "fd": {
        "binary_name": "fd",
        "system_names": ["fd", "fdfind"],
    },
    "rg": {
        "binary_name": "rg",
        "system_names": ["rg"],
    },
}


def get_tool_path(tool: str) -> str | None:
    """检查工具是否在系统 PATH 中可用。"""
    config = _TOOLS.get(tool)
    if not config:
        return None

    for name in config["system_names"]:
        found = shutil.which(name)
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

    config = _TOOLS.get(tool)
    if not config:
        return None

    if not silent:
        logger.warning(
            "%s not found in PATH. Please install it manually.",
            config["binary_name"],
        )

    return None
