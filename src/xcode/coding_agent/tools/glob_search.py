"""基于 ripgrep（优先）/ Python 的文件搜索和目录列取工具。"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from xcode.harness.skills import ToolInput, ToolOutput, ToolSpec
from . import _search_utils
from .path_utils import resolve_absolute_path, matches_blocked_pattern, display_path

MAX_GLOB_RESULTS = 200
MAX_LS_ENTRIES = 500


def build_glob_tools(
    project_root: Path,
    cancel_event: threading.Event | None = None,
) -> tuple[ToolSpec, ...]:
    root = project_root.resolve()

    def _cancel_check() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Tool cancelled")

    def glob_files(data: ToolInput) -> str:
        _cancel_check()
        pattern = str(data.get("pattern", "*")).strip() or "*"
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = _validated_int(data, "max_results", MAX_GLOB_RESULTS, minimum=1)
        return _glob_files(
            root, base, pattern, max_results, _search_utils.get_rg_path()
        )

    def find_files(data: ToolInput) -> str:
        _cancel_check()
        pattern = str(data.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern is required")
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = _validated_int(data, "max_results", MAX_GLOB_RESULTS, minimum=1)
        return _find_files(
            root, base, pattern, max_results, _search_utils.get_rg_path()
        )

    def list_dir(data: ToolInput) -> str:
        _cancel_check()
        raw_path = str(data.get("path", ".")).strip()
        base = _safe_path(root, raw_path)
        limit = _validated_int(data, "limit", MAX_LS_ENTRIES, minimum=1)
        return _ls(root, base, limit)

    return (
        ToolSpec(
            name="glob_files",
            description="Find project files by glob pattern. Use **/*.py for recursive search.",
            input_hint='JSON: {"path": ".", "pattern": "**/*.py", "max_results": 100}',
            handler=glob_files,
            read_only=True,
            concurrency_safe=True,
            group="core",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match, e.g. '*.ts' or 'src/**/*.py'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default: 200)",
                        "minimum": 1,
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="find_files",
            description=(
                "Find files recursively by name or path glob. "
                "Basename-only patterns (e.g. '*.py') automatically match at any depth."
            ),
            input_hint='JSON: {"path": ".", "pattern": "*.py", "max_results": 100}',
            handler=find_files,
            read_only=True,
            concurrency_safe=True,
            group="core",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match, e.g. '*.ts' or 'src/**/*.py'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default: 200)",
                        "minimum": 1,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="list_dir",
            description="List directory contents. Entries sorted alphabetically, '/' suffix for directories. Includes dotfiles.",
            input_hint='JSON: {"path": "src/xcode", "limit": 100}',
            handler=list_dir,
            read_only=True,
            concurrency_safe=True,
            group="core",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries (default: 500)",
                        "minimum": 1,
                    },
                },
                "additionalProperties": False,
            },
        ),
    )


def _glob_files(
    root: Path, base: Path, pattern: str, max_results: int, rg: str | None
) -> str:
    if rg:
        return _glob_with_rg(root, base, pattern, max_results, rg)
    return _glob_with_python(root, base, pattern, max_results, recursive_basename=False)


def _find_files(
    root: Path, base: Path, pattern: str, max_results: int, rg: str | None
) -> str:
    normalized = pattern.replace("\\", "/").removeprefix("./")
    if "/" not in normalized:
        normalized = f"**/{normalized}"

    if rg:
        return _glob_with_rg(root, base, normalized, max_results, rg)
    return _glob_with_python(
        root, base, normalized, max_results, recursive_basename=True
    )


def _glob_with_rg(
    root: Path, base: Path, pattern: str, max_results: int, rg: str
) -> str:
    command = [
        rg,
        "--files",
        "--color",
        "never",
        "--no-require-git",
        "--no-ignore-dot",
        "--no-ignore-exclude",
        "--no-ignore-global",
    ]
    for exclude in (
        "!**/.git/**",
        "!**/.venv/**",
        "!**/__pycache__/**",
        "!**/.local/chroma_db/**",
        "!**/.env",
        "!**/.env.*",
        "!**/.*",
    ):
        command.extend(["--glob", exclude])
    command.extend(["--glob", pattern, "--", str(base)])

    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ValueError(f"ripgrep file discovery failed: {detail}")

    files: list[Path] = []
    for line in completed.stdout.splitlines():
        raw_path = line.strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_file():
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            files.append(path)

    files.sort(key=lambda p: p.as_posix().lower())
    matches = [display_path(root, p) for p in files[:max_results]]
    truncated = len(files) > max_results

    if not matches:
        return ToolOutput("No files found.", metadata={"count": 0, "truncated": False})

    result = "\n".join(matches)
    if truncated:
        result += "\n... truncated"

    return ToolOutput(
        result,
        metadata={"count": len(files), "truncated": truncated},
    )


def _glob_with_python(
    root: Path,
    base: Path,
    pattern: str,
    max_results: int,
    *,
    recursive_basename: bool,
) -> str:
    try:
        files = _search_utils.enumerate_search_files(root, base, use_ripgrep=False)
    except FileNotFoundError as e:
        raise ValueError(str(e))

    matcher = _search_utils.build_path_matcher(
        pattern, recursive_basename=recursive_basename
    )
    matching_files = [
        path
        for path in files
        if matcher(path.relative_to(base).as_posix() if base.is_dir() else path.name)
    ]
    matching_files.sort(
        key=lambda p: (-_search_utils._mtime_ns(p), _search_utils._display(root, p))
    )
    matches = [
        _search_utils._display(root, path) for path in matching_files[:max_results]
    ]

    if not matches:
        return ToolOutput("No files found.", metadata={"count": 0, "truncated": False})

    truncated = len(matching_files) > max_results
    result = "\n".join(matches)
    if truncated:
        result += "\n... truncated"

    return ToolOutput(
        result,
        metadata={"count": len(matching_files), "truncated": truncated},
    )


def _ls(root: Path, base: Path, limit: int) -> str:
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {display_path(root, base)}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {display_path(root, base)}")
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"Permission denied: {display_path(root, base)}") from exc

    lines: list[str] = []
    for entry in entries:
        if len(lines) >= limit:
            break
        if matches_blocked_pattern(entry):
            continue
        name = display_path(root, entry)
        if entry.is_dir():
            name += "/"
        lines.append(name)

    if not lines:
        return ToolOutput(
            "(empty directory)", metadata={"count": 0, "truncated": False}
        )

    result = "\n".join(lines)
    truncated = len(entries) > limit
    if truncated:
        result += f"\n... {len(entries) - limit} more entries omitted"

    return ToolOutput(
        result,
        metadata={"count": len(lines), "truncated": truncated},
    )


def _safe_path(root: Path, raw_path: str) -> Path:
    path = resolve_absolute_path(root, raw_path)
    if matches_blocked_pattern(path):
        raise ValueError(f"path is blocked: {display_path(root, path)}")
    return path


def _validated_int(
    data: dict[str, Any], key: str, default: int, *, minimum: int
) -> int:
    raw = data.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if raw < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return raw
