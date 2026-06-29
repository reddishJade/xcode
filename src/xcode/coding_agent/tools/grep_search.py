"""基于 ripgrep（优先）/ Python 的内容搜索工具。"""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from xcode.harness.skills import ToolInput, ToolOutput, ToolSpec
from . import _search_utils
from .path_utils import resolve_absolute_path, matches_blocked_pattern
from .truncate import GREP_MAX_LINE_LENGTH, truncate_line, truncate_tail

MAX_GREP_RESULTS = 100


def build_grep_tool(
    project_root: Path,
    cancel_event: threading.Event | None = None,
) -> ToolSpec:
    root = project_root.resolve()

    def handler(data: ToolInput) -> str:
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Tool cancelled")
        return _grep(root, _search_utils.get_rg_path(), data)

    return ToolSpec(
        name="grep_search",
        description=(
            "Fast content search tool that works with any codebase size. "
            "Searches file contents using regular expressions. "
            "Supports full regex syntax (eg. 'log.*Error', 'function\\s+\\w+', etc.). "
            "Uses ripgrep when available, falls back to Python grep."
        ),
        input_hint='JSON: {"pattern": "ToolSpec", "path": "src/xcode", "glob": "*.py"}',
        handler=handler,
        read_only=True,
        concurrency_safe=True,
        group="core",
        schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in. Defaults to the project root.",
                },
                "glob": {
                    "type": "string",
                    "description": 'File pattern to include in the search (e.g. "*.py", "*.{ts,tsx}")',
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 100)",
                    "minimum": 1,
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex (default: false)",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of surrounding context to show (default: 0)",
                    "minimum": 0,
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    )


def _grep(root: Path, rg: str | None, data: dict[str, Any]) -> str:
    pattern = str(data.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern is required")

    raw_path = str(data.get("path", ".")).strip()
    if raw_path:
        filepath = resolve_absolute_path(root, raw_path)
        if matches_blocked_pattern(filepath):
            raise ValueError(
                f"path is blocked: {_search_utils._display(root, filepath)}"
            )

    glob_pattern = str(data.get("glob")) if data.get("glob") else None
    max_results = _validated_int(data, "max_results", MAX_GREP_RESULTS, minimum=1)
    ignore_case = bool(data.get("ignore_case", False))
    literal = bool(data.get("literal", False))
    context = _validated_int(data, "context", 0, minimum=0)

    if rg:
        return _grep_with_rg(
            root,
            raw_path,
            pattern,
            glob_pattern,
            max_results,
            ignore_case,
            literal,
            context,
            rg,
        )
    return _grep_fallback(
        root,
        raw_path,
        pattern,
        glob_pattern,
        max_results,
        ignore_case,
        literal,
        context,
    )


def _grep_with_rg(
    root: Path,
    raw_path: str,
    pattern: str,
    glob_pattern: str | None,
    max_results: int,
    ignore_case: bool,
    literal: bool,
    context: int,
    rg: str,
) -> str:
    command = [
        rg,
        "--line-number",
        "--no-heading",
        "--with-filename",
        "--color",
        "never",
        "--no-require-git",
        "--no-ignore-dot",
        "--no-ignore-exclude",
        "--no-ignore-global",
    ]
    if ignore_case:
        command.append("--ignore-case")
    if literal:
        command.append("--fixed-strings")
    if context > 0:
        command.extend(["--context", str(context)])
    if glob_pattern:
        command.extend(["--glob", glob_pattern])
    command.extend(_search_utils._rg_exclusion_args())
    command.extend(["--", pattern])

    search_path = resolve_absolute_path(root, raw_path) if raw_path else root
    command.append(str(search_path))

    proc = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    lines: list[str] = []
    truncated_lines = 0
    total = 0
    stopped_for_limit = False
    stderr = ""
    try:
        while True:
            raw_line = proc.stdout.readline()
            if not raw_line:
                break
            line = raw_line.rstrip("\n").rstrip("\r")
            truncated, was_truncated = truncate_line(line)
            if was_truncated:
                truncated_lines += 1
            lines.append(truncated)
            total += 1
            if total >= max_results:
                stopped_for_limit = True
                proc.kill()
                break
        proc.wait(timeout=5)
        stderr = proc.stderr.read().strip()
    finally:
        proc.stdout.close()
        proc.stderr.close()

    if not stopped_for_limit and proc.returncode not in (0, 1):
        detail = stderr or f"exit code {proc.returncode}"
        raise ValueError(f"ripgrep failed: {detail}")

    if not lines:
        return ToolOutput(
            "No matches found.", metadata={"matches": 0, "truncated": False}
        )

    result = "\n".join(lines)
    if truncated_lines > 0:
        result += f"\n[Truncated {truncated_lines} long lines to {GREP_MAX_LINE_LENGTH} chars]"

    tr = truncate_tail(result)
    truncated = stopped_for_limit or tr.truncated
    if tr.truncated:
        result = tr.content
        result += (
            f"\n[Showing {tr.output_lines} of {tr.total_lines} lines "
            f"({tr.max_bytes // 1024}KB limit). "
            f"Use 'max_results=N*2' for more, or refine pattern.]"
        )

    return ToolOutput(
        result,
        metadata={"matches": total, "truncated": truncated},
    )


def _grep_fallback(
    root: Path,
    raw_path: str,
    pattern: str,
    glob_pattern: str | None,
    max_results: int,
    ignore_case: bool,
    literal: bool,
    context: int,
) -> str:
    base = resolve_absolute_path(root, raw_path) if raw_path else root
    files = _search_utils.enumerate_search_files(root, base, use_ripgrep=False)
    if glob_pattern:
        matcher = _search_utils.build_path_matcher(
            glob_pattern, recursive_basename=True
        )
        files = [
            path
            for path in files
            if matcher(
                path.relative_to(base).as_posix() if base.is_dir() else path.name
            )
        ]

    re_pattern: re.Pattern[str] | None = None
    if not literal:
        try:
            flags = re.IGNORECASE if ignore_case else 0
            re_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

    matches: list[str] = []
    for path in files:
        if len(matches) >= max_results:
            break
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        all_lines = text.splitlines()
        match_indices: list[int] = []
        for line_no, line in enumerate(all_lines, 1):
            if literal:
                search_line = line.casefold() if ignore_case else line
                search_pattern = pattern.casefold() if ignore_case else pattern
                matched = search_pattern in search_line
            else:
                matched = bool(re_pattern.search(line)) if re_pattern else False
            if matched:
                match_indices.append(line_no)
                if len(matches) >= max_results:
                    break
        for line_no in match_indices:
            if len(matches) >= max_results:
                break
            start = max(0, line_no - 1 - context) if context > 0 else line_no - 1
            end = min(len(all_lines), line_no + context) if context > 0 else line_no
            for ctx_line_no in range(start, end):
                prefix = ">" if ctx_line_no + 1 == line_no else "-"
                raw = all_lines[ctx_line_no]
                truncated_raw, _ = truncate_line(raw)
                matches.append(
                    f"{_search_utils._display(root, path)}:{ctx_line_no + 1}:{prefix} {truncated_raw}"
                )

    total = len(matches)
    if not matches:
        return ToolOutput(
            "No matches found.", metadata={"matches": 0, "truncated": False}
        )

    result = "\n".join(matches)
    tr = truncate_tail(result)
    truncated = tr.truncated
    if tr.truncated:
        result = tr.content
        result += (
            f"\n[Showing {tr.output_lines} of {tr.total_lines} lines "
            f"({tr.max_bytes // 1024}KB limit). "
            f"Use 'max_results=N*2' for more, or refine pattern.]"
        )

    return ToolOutput(
        result,
        metadata={"matches": total, "truncated": truncated},
    )


def _validated_int(
    data: dict[str, Any], key: str, default: int, *, minimum: int
) -> int:
    raw = data.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if raw < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return raw
