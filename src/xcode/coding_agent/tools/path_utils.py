from __future__ import annotations

import unicodedata
from pathlib import Path

from xcode.harness.skills import resolve_project_path
from .truncate import truncate_tail

import pathspec

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024

_BLOCKED_SPEC = pathspec.PathSpec.from_lines(
    "gitwildmatch",
    [
        ".git/",
        ".venv/",
        "__pycache__/",
        ".env",
        ".local/chroma_db/",
    ],
)


def is_path_blocked(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True
    return _BLOCKED_SPEC.match_file(relative.as_posix())


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

    narrow_space = str(resolved).replace("\u202f", " ")
    if narrow_space != str(resolved):
        narrow_path = Path(narrow_space)
        if narrow_path.exists():
            return narrow_path

        nfd_narrow = unicodedata.normalize("NFD", narrow_space)
        if nfd_narrow != str(resolved):
            nfd_narrow_path = Path(nfd_narrow)
            if nfd_narrow_path.exists():
                return nfd_narrow_path

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
