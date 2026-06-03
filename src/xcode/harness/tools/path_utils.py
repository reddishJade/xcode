"""文件路径安全约束和输出截断共享工具。"""

from __future__ import annotations

from pathlib import Path

BLOCKED_PARTS = {".git", ".venv", "__pycache__"}

MAX_RETURN_CHARS = 50_000


def is_path_blocked(root: Path, path: Path) -> bool:
    """判断 path 是否在受保护目录内或逃逸出 root。"""
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True
    parts = set(relative.parts)
    if parts & BLOCKED_PARTS:
        return True
    if ".env" in relative.parts or relative.name == ".env":
        return True
    if (
        len(relative.parts) >= 2
        and relative.parts[0] == ".local"
        and relative.parts[1] == "chroma_db"
    ):
        return True
    return (
        len(relative.parts) >= 3
        and relative.parts[0] == "xcode"
        and relative.parts[1] == ".local"
        and relative.parts[2] == "chroma_db"
    )


def truncate_output(text: str, max_chars: int = MAX_RETURN_CHARS) -> str:
    """截断过长文本，保留首尾并在中间标注被截断的字符数。"""
    if len(text) <= max_chars:
        return text
    keep = (max_chars - 80) // 2
    return (
        text[:keep]
        + f"\n\n[... truncated {len(text) - keep * 2} chars ...]\n\n"
        + text[-keep:]
    )


def display_path(root: Path, path: Path) -> str:
    """将绝对路径转为相对于 root 的 POSIX 路径，逃逸时回退为绝对路径。"""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)