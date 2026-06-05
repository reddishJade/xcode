from __future__ import annotations


from difflib import unified_diff
import json
from pathlib import Path
from typing import Any, Protocol

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.skills import ToolInput, ToolOutput, ToolSpec, resolve_project_path
from .edit_diff import (
    apply_edits_fuzzy,
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation
from .path_utils import (
    is_path_blocked,
    resolve_read_path,
    truncate_output,
    display_path,
)

"""受沙箱约束的本地文件工具。

文件工具负责读写项目内文本文件，并在工具层集中处理路径解析、敏感目录
拒绝、输出截断和基于 old_text 的文件编辑（含模糊匹配）。
"""

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000


class FileOperations(Protocol):
    def exists(self, path: Path) -> bool: ...

    def is_file(self, path: Path) -> bool: ...

    def is_dir(self, path: Path) -> bool: ...

    def size(self, path: Path) -> int: ...

    def read_bytes(self, path: Path) -> bytes: ...

    def write_bytes(self, path: Path, data: bytes) -> None: ...

    def mkdir(self, path: Path) -> None: ...


class LocalFileOperations:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_file(self, path: Path) -> bool:
        return path.is_file()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def size(self, path: Path) -> int:
        return path.stat().st_size

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        path.write_bytes(data)

    def mkdir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)


READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Project-relative text file path.",
        },
        "limit": {
            "type": "integer",
            "description": "Optional max number of lines to return.",
        },
        "offset": {
            "type": "integer",
            "description": "Optional 1-based line number to start reading from.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}

WRITE_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Project-relative file path to write.",
        },
        "content": {
            "type": "string",
            "description": "Full UTF-8 file content.",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

EDIT_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Project-relative file path to edit.",
        },
        "edits": {
            "type": "array",
            "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally.",
            "items": {
                "type": "object",
                "properties": {
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find. Must match exactly one occurrence.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def build_file_tools(
    project_root: Path,
    context_state: ContextualRetrievalState | None = None,
    operations: FileOperations | None = None,
) -> tuple[ToolSpec, ...]:
    root = project_root.resolve()
    ops = operations or LocalFileOperations()

    return (
        ToolSpec(
            name="read_file",
            description="Read a text file inside the project sandbox.",
            input_hint='JSON: {"path": "src/xcode/main.py", "offset": 1, "limit": 80}',
            handler=lambda data: _read_file(root, ops, context_state, data),
            risk="low",
            schema=READ_FILE_SCHEMA,
            read_only=True,
            concurrency_safe=True,
            group="core",
            prompt_snippet="Read a text file inside the project sandbox",
            prompt_guidelines=(
                "Use read_file offset and limit to continue reading long files.",
            ),
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Create a new text file or intentionally replace an entire file inside "
                "the project sandbox. Prefer edit_file for targeted changes to an "
                "existing file."
            ),
            input_hint='JSON: {"path": "notes.md", "content": "..."}',
            handler=lambda data: _write_file(root, ops, context_state, data),
            risk="high",
            schema=WRITE_FILE_SCHEMA,
            group="core",
            counts_as_progress=True,
            prompt_snippet="Create new files or deliberately replace entire files",
            prompt_guidelines=(
                "Use write_file only for new files or deliberate full-file rewrites.",
            ),
            examples=[
                {
                    "path": "notes.md",
                    "content": "# Notes\n\nNew file content.\n",
                }
            ],
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Modify an existing text file with one or more targeted replacements. "
                "Use this tool when changing existing code or docs so unrelated "
                "content is preserved."
            ),
            input_hint='JSON: {"path": "src/main.py", "edits": [{"old_text": "...", "new_text": "..."}]}',
            handler=lambda data: _edit_file(root, ops, context_state, data),
            risk="high",
            schema=EDIT_FILE_SCHEMA,
            group="core",
            counts_as_progress=True,
            prompt_snippet=(
                "Make precise file edits with exact old_text/new_text replacements"
            ),
            prompt_guidelines=(
                "Use edit_file for precise changes to existing files.",
                "When changing multiple separate locations in one file, use one edit_file call with multiple entries in edits.",
                "Each edit_file edits[].old_text is matched against the original file, not after earlier edits are applied.",
                "Keep edit_file edits[].old_text as small as possible while still unique in the file.",
                "Do not emit overlapping or nested edit_file edits; merge nearby changes into one edit.",
            ),
            examples=[
                {
                    "path": "src/main.py",
                    "edits": [
                        {
                            "old_text": "return old_value\n",
                            "new_text": "return new_value\n",
                        }
                    ],
                }
            ],
        ),
    )


def _read_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        raise ValueError("path is required")
    path = resolve_read_path(root, path_str)
    if is_path_blocked(root, path):
        raise ValueError(f"path is blocked: {_display(root, path)}")
    if not operations.is_file(path):
        raise ValueError(f"not a file: {_display(root, path)}")
    size = operations.size(path)
    if size > MAX_READ_BYTES:
        raise ValueError(f"file too large: {size} bytes")
    text, _encoding = _read_text(path, operations)
    if context_state is not None:
        context_state.record_file(path)
    offset = data.get("offset")
    limit = data.get("limit")
    if offset is not None or limit is not None:
        text = _select_lines(text, _display(root, path), offset, limit)
    return _truncate(text)


def _write_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        raise ValueError("path is required")
    path = _safe_path(root, path_str)
    if operations.exists(path) and operations.is_dir(path):
        raise ValueError(f"path is a directory: {_display(root, path)}")
    if "content" not in data:
        raise ValueError("content is required")
    content = str(data.get("content", ""))
    _ensure_write_size(content)
    return with_file_mutation(
        path, lambda: _write_file_impl(root, path, content, operations, context_state)
    )


def _write_file_impl(
    root: Path,
    path: Path,
    content: str,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
) -> str:
    operations.mkdir(path.parent)
    _write_text(path, content, "utf-8", operations)
    if context_state is not None:
        context_state.record_file(path)
    return f"wrote file: {_display(root, path)}"


def _edit_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        raise ValueError("path is required")
    path = _safe_path(root, path_str)
    if operations.exists(path) and operations.is_dir(path):
        raise ValueError(f"path is a directory: {_display(root, path)}")
    if not operations.is_file(path):
        raise ValueError(f"not a file: {_display(root, path)}")

    edits = _prepare_edits(data)
    if not edits:
        raise ValueError("no edits provided")

    return with_file_mutation(
        path,
        lambda: _edit_file_impl(root, path, edits, data, operations, context_state),
    )


def _edit_file_impl(
    root: Path,
    path: Path,
    edits: list[dict[str, str]],
    data: ToolInput,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
) -> str:
    content, encoding = _read_text(path, operations)
    bom, clean_content = strip_bom(content)
    ending = detect_line_ending(clean_content)
    normalized = normalize_to_lf(clean_content)

    replace_all = bool(data.get("replace_all", False))
    if replace_all and len(edits) == 1:
        count = normalized.count(edits[0]["old_text"])
        if count == 0:
            raise ValueError("old_text not found")
        updated = normalized.replace(edits[0]["old_text"], edits[0]["new_text"])
        edit_count = count
    else:
        updated, edit_count = apply_edits_fuzzy(normalized, edits, _display(root, path))

    restored = restore_line_endings(updated, ending)
    final = bom + restored
    _ensure_write_size(final)
    _write_text(path, final, encoding, operations)
    if context_state is not None:
        context_state.record_file(path)
    diff = _diff_preview(_display(root, path), content, final)
    return ToolOutput(
        f"edited file: {_display(root, path)} replacements={edit_count}\n{diff}",
        metadata={
            "patch": diff,
            "first_changed_line": _first_changed_line(content, final),
        },
    )


def _parse_optional_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _select_lines(
    text: str,
    display_path: str,
    offset: Any,
    limit: Any,
) -> str:
    offset_value = 1 if offset is None else _parse_optional_int(offset, "offset")
    if offset_value < 1:
        raise ValueError("offset must be positive")

    limit_value = None if limit is None else _parse_optional_int(limit, "limit")
    if limit_value is not None and limit_value < 0:
        raise ValueError("limit must be non-negative")

    lines = text.splitlines()
    if not lines:
        return ""
    start = offset_value - 1
    if start >= len(lines):
        raise ValueError(
            f"offset {offset_value} is beyond end of file ({len(lines)} lines total)"
        )
    end = len(lines) if limit_value is None else min(start + limit_value, len(lines))
    selected = "\n".join(lines[start:end])
    if end < len(lines):
        continuation: dict[str, int | str] = {
            "path": display_path,
            "offset": end + 1,
        }
        if limit_value is not None:
            continuation["limit"] = limit_value
        selected += (
            f"\n\n[Showing lines {offset_value}-{end} of {len(lines)}. "
            f"Use read_file with {json.dumps(continuation, ensure_ascii=False)} "
            "to continue.]"
        )
    return selected


def read_project_text_file(project_root: Path, raw_path: str) -> str:
    operations = LocalFileOperations()
    root = project_root.resolve()
    path = _safe_path(root, raw_path)
    if not operations.is_file(path):
        raise ValueError(f"not a file: {_display(root, path)}")
    size = operations.size(path)
    if size > MAX_READ_BYTES:
        raise ValueError(f"file too large: {size} bytes")
    text, _encoding = _read_text(path, operations)
    return _truncate(text)


def _safe_path(root: Path, raw_path: str) -> Path:
    path = resolve_project_path(root, raw_path)
    if is_path_blocked(root, path):
        raise ValueError(f"path is blocked: {display_path(root, path)}")
    return path


def _truncate(text: str) -> str:
    return truncate_output(text)


def _read_text(path: Path, operations: FileOperations) -> tuple[str, str]:
    data = operations.read_bytes(path)
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        return data.decode(encoding), encoding
    except UnicodeDecodeError:
        pass
    raise UnicodeDecodeError("utf-8", data, 0, 1, "file is not valid UTF-8 text")


def _write_text(
    path: Path,
    text: str,
    encoding: str,
    operations: FileOperations,
) -> None:
    operations.write_bytes(path, text.encode(encoding))


def _ensure_write_size(text: str) -> None:
    size = len(text.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        raise ValueError(f"write content too large: {size} bytes")


def _diff_preview(path: str, before: str, after: str) -> str:
    lines = unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(lines)


def _first_changed_line(before: str, after: str) -> int | None:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    for index, (before_line, after_line) in enumerate(
        zip(before_lines, after_lines), start=1
    ):
        if before_line != after_line:
            return index
    if len(before_lines) != len(after_lines):
        return min(len(before_lines), len(after_lines)) + 1
    return None


def _display(root: Path, path: Path) -> str:
    return display_path(root, path)


def _prepare_edits(data: dict[str, Any]) -> list[dict[str, str]]:
    """将输入归一化为 edits 列表。"""
    edits: list[dict[str, str]] = []

    raw_edits = data.get("edits")
    if isinstance(raw_edits, list):
        for i, edit in enumerate(raw_edits):
            if not isinstance(edit, dict):
                raise ValueError(f"edits[{i}]: must be an object")
            old = str(edit.get("old_text", ""))
            new = str(edit.get("new_text", ""))
            if not old:
                raise ValueError(f"edits[{i}].old_text must not be empty")
            edits.append({"old_text": old, "new_text": new})

    old_text = str(data.get("old_text", "")).strip()
    if old_text:
        if "new_text" not in data:
            raise ValueError("new_text is required")
        edits.append({"old_text": old_text, "new_text": str(data.get("new_text", ""))})

    return edits
