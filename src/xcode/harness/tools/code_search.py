from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ..skills import ToolSpec, parse_tool_input, resolve_project_path
from .file import MAX_RETURN_CHARS

"""供编码 Agent 使用的只读代码搜索工具。"""

MAX_GREP_RESULTS = 100
MAX_GLOB_RESULTS = 200
MAX_LS_ENTRIES = 500
BLOCKED_PARTS = {".git", ".venv", "__pycache__"}
_RG_MISSING_HINT_EMITTED = False


def build_code_tools(project_root: Path) -> tuple[ToolSpec, ...]:
    root = project_root.resolve()

    def grep_search(action_input: str) -> str:
        data = parse_tool_input(action_input, default_key="pattern")
        pattern = str(data.get("pattern", "")).strip()
        if not pattern:
            return "pattern is required"
        base = _safe_path(root, str(data.get("path", ".")))
        glob = data.get("glob")
        max_results = int(data.get("max_results", MAX_GREP_RESULTS))
        return _grep(root, base, pattern, str(glob) if glob else None, max_results)

    def glob_files(action_input: str) -> str:
        data = parse_tool_input(action_input, default_key="pattern")
        pattern = str(data.get("pattern", "*")).strip() or "*"
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = int(data.get("max_results", MAX_GLOB_RESULTS))
        return _glob_files(root, base, pattern, max_results)

    def ls_files(action_input: str) -> str:
        data = parse_tool_input(action_input)
        raw_path = str(data.get("path", ".")).strip()
        base = _safe_path(root, raw_path)
        limit = int(data.get("limit", MAX_LS_ENTRIES))
        return _ls(root, base, limit)

    return (
        ToolSpec(
            name="glob_files",
            description="Find project files by glob pattern. Use **/*.py for recursive search.",
            input_hint='JSON: {"path": ".", "pattern": "**/*.py", "max_results": 100}',
            handler=glob_files,
            risk="low",
            read_only=True,
            concurrency_safe=True,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
            },
        ),
        ToolSpec(
            name="grep_search",
            description="Search exact project text. Uses ripgrep when available, then falls back to Python grep.",
            input_hint='JSON: {"pattern": "ToolSpec", "path": "src/xcode", "glob": "*.py"}',
            handler=grep_search,
            risk="low",
            read_only=True,
            concurrency_safe=True,
        ),
        ToolSpec(
            name="ls",
            description="List directory contents. Entries sorted alphabetically, '/' suffix for directories. Includes dotfiles.",
            input_hint='JSON: {"path": "src/xcode", "limit": 100}',
            handler=ls_files,
            risk="low",
            read_only=True,
            concurrency_safe=True,
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
                    },
                },
                "additionalProperties": False,
            },
        ),
    )


def _grep(
    root: Path,
    base: Path,
    pattern: str,
    glob: str | None,
    max_results: int,
) -> str:
    global _RG_MISSING_HINT_EMITTED
    rg = shutil.which("rg")
    if rg:
        command = [rg, "--line-number", "--no-heading", "--color", "never", pattern]
        if glob:
            command.extend(["-g", glob])
        command.append(str(base))
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
        output = completed.stdout.strip()
        if not output:
            return "No matches found."
        lines = output.splitlines()[:max_results]
        return _truncate("\n".join(lines))
    hint = ""
    if not _RG_MISSING_HINT_EMITTED:
        _RG_MISSING_HINT_EMITTED = True
        hint = "[ripgrep not found; install rg for faster search. Falling back to Python grep.]\n"
    return hint + _grep_fallback(root, base, pattern, glob, max_results)


def _glob_files(root: Path, base: Path, pattern: str, max_results: int) -> str:
    files = sorted(base.glob(pattern) if base.is_dir() else [base])
    matches: list[str] = []
    for path in files:
        if len(matches) >= max_results:
            break
        if path.is_file() and not _is_blocked(root, path):
            matches.append(_display(root, path))
    if not matches:
        return "No files found."
    suffix = "\n... truncated" if len(matches) >= max_results else ""
    return "\n".join(matches) + suffix


def _ls(root: Path, base: Path, limit: int) -> str:
    if not base.exists():
        return f"Path not found: {_display(root, base)}"
    if not base.is_dir():
        return f"Not a directory: {_display(root, base)}"
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        return f"Permission denied: {_display(root, base)}"

    lines: list[str] = []
    for entry in entries:
        if len(lines) >= limit:
            break
        if _is_blocked(root, entry):
            continue
        name = _display(root, entry)
        if entry.is_dir():
            name += "/"
        lines.append(name)

    if not lines:
        return "(empty directory)"
    result = "\n".join(lines)
    if len(entries) > limit:
        result += f"\n... {len(entries) - limit} more entries omitted"
    return result


def _grep_fallback(
    root: Path,
    base: Path,
    pattern: str,
    glob: str | None,
    max_results: int,
) -> str:
    files = sorted(base.rglob(glob or "*")) if base.is_dir() else [base]
    matches: list[str] = []
    for path in files:
        if len(matches) >= max_results:
            break
        if not path.is_file() or _is_blocked(root, path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            if pattern in line:
                matches.append(f"{_display(root, path)}:{line_no}:{line}")
                if len(matches) >= max_results:
                    break
    return _truncate("\n".join(matches)) if matches else "No matches found."


def _safe_path(root: Path, raw_path: str) -> Path:
    path = resolve_project_path(root, raw_path)
    if _is_blocked(root, path):
        raise ValueError(f"path is blocked: {_display(root, path)}")
    return path


def _is_blocked(root: Path, path: Path) -> bool:
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


def _truncate(text: str) -> str:
    if len(text) <= MAX_RETURN_CHARS:
        return text
    keep = (MAX_RETURN_CHARS - 80) // 2
    return (
        text[:keep]
        + f"\n\n[... truncated {len(text) - keep * 2} chars ...]\n\n"
        + text[-keep:]
    )


def _display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)
