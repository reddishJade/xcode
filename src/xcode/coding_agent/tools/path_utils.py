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
        "**/.local/chroma_db/",
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


_BINARY_EXTENSIONS: set[str] = {
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".class",
    ".jar",
    ".war",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".bin",
    ".dat",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".wasm",
    ".pyc",
    ".pyo",
}

SAMPLE_BYTES = 4096


def is_binary_file(path: Path, sample: bytes) -> bool:
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    if not sample:
        return False
    non_printable = 0
    for b in sample:
        if b == 0:
            return True
        if b < 9 or (b > 13 and b < 32):
            non_printable += 1
    return non_printable / len(sample) > 0.3


def resolve_absolute_path(root: Path, raw_path: str) -> Path:
    path_str = raw_path.strip().strip("\"'")
    if not path_str:
        raise ValueError("path is required")
    p = Path(path_str)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        if ".." in p.parts:
            raise ValueError("parent-directory paths are not allowed")
        resolved = (root / p).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError("path escapes project root")
    return resolved


def matches_blocked_pattern(path: Path) -> bool:
    try:
        return _BLOCKED_SPEC.match_file(path.resolve().as_posix())
    except OSError:
        return False
