"""暴露给 Agent 的工作区工具。"""

from .bash import build_bash_tool
from .code_search import build_code_tools
from .file import build_file_tools
from .shell_adapter import ShellSpec, detect_shell, build_shell_argv
from .tools_manager import ensure_tool
from .plan_mode import build_plan_mode_tools
from .worktree import WorktreeTaskRunner, build_worktree_tools

__all__ = [
    "build_bash_tool",
    "build_code_tools",
    "build_file_tools",
    "build_plan_mode_tools",
    "build_worktree_tools",
    "ensure_tool",
    "ShellSpec",
    "WorktreeTaskRunner",
    "detect_shell",
    "build_shell_argv",
]
