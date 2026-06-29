"""暴露给 Agent 的工作区工具。"""

from .apply_patch import build_apply_patch_tool
from .bash import build_bash_tool
from .file import build_file_tools
from .glob_search import build_glob_tools
from .grep_search import build_grep_tool
from .shell_adapter import ShellSpec, detect_shell, build_shell_argv
from .tools_manager import ensure_tool

__all__ = [
    "build_apply_patch_tool",
    "build_bash_tool",
    "build_file_tools",
    "build_glob_tools",
    "build_grep_tool",
    "ensure_tool",
    "ShellSpec",
    "detect_shell",
    "build_shell_argv",
]
