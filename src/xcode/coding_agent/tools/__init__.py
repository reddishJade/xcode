"""暴露给 Agent 的工作区工具。"""

from .apply_patch import build_apply_patch_tool
from .bash import build_bash_tool
from .glob_search import build_glob_tools
from .grep_search import build_grep_tool
from .question import build_question_tool
from .read_file import build_read_file_tool
from .shell_adapter import ShellSpec, build_shell_argv, detect_shell
from .subagent import build_subagent_tools
from .todowrite import build_todowrite_tool
from .tools_manager import ensure_tool
from .webfetch import build_webfetch_tool
from .websearch import build_websearch_tool
from .write_file import build_write_file_tools

__all__ = [
    "ShellSpec",
    "build_apply_patch_tool",
    "build_bash_tool",
    "build_glob_tools",
    "build_grep_tool",
    "build_question_tool",
    "build_read_file_tool",
    "build_shell_argv",
    "build_subagent_tools",
    "build_todowrite_tool",
    "build_webfetch_tool",
    "build_websearch_tool",
    "build_write_file_tools",
    "detect_shell",
    "ensure_tool",
]
