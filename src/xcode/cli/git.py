"""CLI/TUI 使用的 Git 工作区信息。"""

from __future__ import annotations

from pathlib import Path
import subprocess


def git_branch_name(project_root: Path) -> str | None:
    """返回当前 Git 分支名；目录不是 Git 工作区时返回 None。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "branch", "--show-current"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None
