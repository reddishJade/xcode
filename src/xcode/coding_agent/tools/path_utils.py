from __future__ import annotations

import unicodedata
from pathlib import Path

from xcode.harness.skills import resolve_project_path
from .truncate import truncate_tail

BLOCKED_PARTS = {".git", ".venv", "__pycache__"}

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024


def is_path_blocked(root: Path, path: Path) -> bool:
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


def resolve_read_path(root: Path, raw_path: str) -> Path:
    """解析路径并尝试 macOS 特有文件名变体 fallback。"""
    resolved = resolve_project_path(root, raw_path)
    if resolved.exists():
        return resolved

    nfd = unicodedata.normalize("NFD", str(resolved))
    if nfd != str(resolved):
        nfd_path = Path(nfd)
        if nfd_path.exists():
            return nfd_path

    curly = str(resolved).replace("'", "\u2019")
    if curly != str(resolved):
        curly_path = Path(curly)
        if curly_path.exists():
            return curly_path

    nfd_curly = unicodedata.normalize("NFD", curly)
    if nfd_curly != str(resolved):
        nfd_curly_path = Path(nfd_curly)
        if nfd_curly_path.exists():
            return nfd_curly_path

    return resolved


def truncate_output(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    return truncate_tail(text, max_lines=max_lines, max_bytes=max_bytes).content


def display_path(root: Path, path: Path) -> str:
    """将绝对路径转为相对于 root 的 POSIX 路径，逃逸时回退为绝对路径。"""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)
