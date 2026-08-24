"""`xcode config` 子命令：打开交互式配置浏览器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_registry import run_config_browser
from .setup_wizard import CONFIG_FILENAME


def handle_config_command(args: Any, project_root: Path) -> None:
    """启动全量设置的交互式浏览器，写入项目配置文件。"""
    config_path = args.config or project_root / CONFIG_FILENAME
    try:
        run_config_browser(config_path)
    except KeyboardInterrupt:
        return
