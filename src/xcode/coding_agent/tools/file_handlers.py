"""受沙箱约束的本地文件工具。

文件工具负责读写项目内文本文件，并在工具层集中处理路径解析、敏感目录
拒绝、输出截断和基于 old_text 的文件编辑（含模糊匹配）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from difflib import unified_diff
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.skills import (
    ToolInput,
    ToolOutput,
    resolve_project_path,
)
from .edit_diff import (
    apply_fuzzy_replace,
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_image import _detect_image, _read_image
from .file_mutation_queue import with_file_mutation
from .path_utils import (
    is_path_blocked,
    truncate_output,
    display_path,
    resolve_absolute_path,
    matches_blocked_pattern,
    is_binary_file,
    SAMPLE_BYTES,
)


MAX_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024
MAX_BYTES_LABEL = "50 KB"
MAX_LINE_SUFFIX = f"... (line truncated to {MAX_LINE_LENGTH} chars)"
MAX_FILE_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000


@dataclass(frozen=True)
class FileEdit:
    old_text: str
    new_text: str


@dataclass(frozen=True)
class EditFileRequest:
    path: Path
    edits: list[FileEdit]
    replace_all: bool


@dataclass(frozen=True)
class WriteFileRequest:
    path: Path
    content: str


class FileOperations(Protocol):
    def exists(self, path: Path) -> bool: ...
    def is_file(self, path: Path) -> bool: ...
    def is_dir(self, path: Path) -> bool: ...
    def size(self, path: Path) -> int: ...
    def read_bytes(self, path: Path) -> bytes: ...
    def write_bytes(self, path: Path, data: bytes) -> None: ...
    def mkdir(self, path: Path) -> None: ...
    def remove_file(self, path: Path) -> None: ...
    def iter_lines(self, path: Path) -> Iterator[str]: ...
    def read_dir_entries(self, path: Path) -> list[tuple[str, bool]]: ...
    def read_head(self, path: Path, n: int) -> bytes: ...


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

    def remove_file(self, path: Path) -> None:
        path.unlink()

    def iter_lines(self, path: Path) -> Iterator[str]:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                yield line.rstrip("\r\n")

    def read_dir_entries(self, path: Path) -> list[tuple[str, bool]]:
        entries = [(entry.name, entry.is_dir()) for entry in path.iterdir()]
        entries.sort(key=lambda x: x[0].casefold())
        return entries

    def read_head(self, path: Path, n: int) -> bytes:
        with open(path, "rb") as f:
            return f.read(n)


def _read_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        raise ValueError("path is required")

    filepath = resolve_absolute_path(root, path_str)
    if matches_blocked_pattern(filepath):
        raise ValueError(f"path is blocked: {_display(root, filepath)}")

    if not operations.exists(filepath):
        _file_not_found(filepath, root, operations)

    if context_state is not None:
        context_state.record_file(filepath)

    display = _display(root, filepath)
    if operations.is_dir(filepath):
        return _read_directory(filepath, display, operations, data)

    mime = _detect_image(filepath, operations)
    if mime is not None:
        return _read_image(filepath, display, mime, operations)

    sample = operations.read_head(filepath, SAMPLE_BYTES)
    if is_binary_file(filepath, sample):
        return ToolOutput(
            f"Cannot read binary file: {display}",
            is_error=True,
        )

    offset = _parse_offset(data)
    limit = _parse_limit(data)
    return _read_text_file(filepath, display, offset, limit, operations)


def _parse_offset(data: ToolInput) -> int:
    raw = data.get("offset")
    if raw is None:
        return 1
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError("offset must be an integer")
    if val < 1:
        raise ValueError("offset must be positive")
    return val


def _parse_limit(data: ToolInput) -> int:
    raw = data.get("limit")
    if raw is None:
        return MAX_READ_LIMIT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    if val < 0:
        raise ValueError("limit must be non-negative")
    return val if val > 0 else MAX_READ_LIMIT


def _file_not_found(filepath: Path, root: Path, operations: FileOperations) -> str:
    dir_path = filepath.parent
    base_name = filepath.stem if filepath.suffix else filepath.name
    suggestions: list[str] = []

    try:
        if operations.is_dir(dir_path):
            for entry_name, _ in operations.read_dir_entries(dir_path):
                if (
                    base_name.lower() in entry_name.lower()
                    or entry_name.lower() in base_name.lower()
                ):
                    suggestions.append(str(dir_path / entry_name))
                    if len(suggestions) >= 3:
                        break
    except OSError:
        pass

    display = _display(root, filepath)
    if suggestions:
        raise ValueError(
            f"File not found: {display}\n\n"
            f"Did you mean one of these?\n"
            f"{chr(10).join(suggestions)}"
        )
    raise ValueError(f"File not found: {display}")


def _truncate_line(line: str) -> str:
    if len(line) > MAX_LINE_LENGTH:
        return line[:MAX_LINE_LENGTH] + MAX_LINE_SUFFIX
    return line


def _read_text_file(
    path: Path,
    display: str,
    offset: int,
    limit: int,
    operations: FileOperations,
) -> ToolOutput:
    start = offset - 1
    lines: list[str] = []
    bytes_count = 0
    total_count = 0
    more = False
    cut = False

    for line in operations.iter_lines(path):
        total_count += 1
        if total_count <= start:
            continue
        if len(lines) >= limit:
            more = True
            continue
        truncated = _truncate_line(line)
        line_bytes = len(truncated.encode("utf-8")) + (1 if lines else 0)
        if bytes_count + line_bytes > MAX_BYTES:
            cut = True
            more = True
            break
        lines.append(truncated)
        bytes_count += line_bytes

    if total_count < offset and not (total_count == 0 and offset == 1):
        raise ValueError(
            f"Offset {offset} is out of range for this file ({total_count} lines)"
        )

    last_line = offset + len(lines) - 1
    next_line = offset if not lines else last_line + 1

    output_parts = [
        f"<path>{display}</path>",
        "<type>file</type>",
        "<content>",
    ]
    for i, line in enumerate(lines):
        output_parts.append(f"{i + offset}: {line}")

    if cut:
        output_parts.append(
            f"\n(Output capped at {MAX_BYTES_LABEL}. "
            f"Showing lines {offset}-{last_line}. "
            f"Use offset={next_line} to continue.)"
        )
    elif more:
        output_parts.append(
            f"\n(Showing lines {offset}-{last_line} of {total_count}. "
            f"Use offset={next_line} to continue.)"
        )
    else:
        output_parts.append(f"\n(End of file - total {total_count} lines)")
    output_parts.append("</content>")
    text = "\n".join(output_parts)

    return ToolOutput(
        text,
        metadata={
            "display": {
                "type": "file",
                "path": display,
                "text": "\n".join(lines),
                "lineStart": offset,
                "lineEnd": last_line,
                "totalLines": total_count,
                "truncated": more or cut,
            },
        },
    )


def _read_directory(
    path: Path,
    display: str,
    operations: FileOperations,
    data: ToolInput,
) -> ToolOutput:
    items = operations.read_dir_entries(path)
    formatted = [name + "/" if is_dir else name for name, is_dir in items]

    offset = data.get("offset", 1)
    limit = data.get("limit", MAX_READ_LIMIT)

    start = offset - 1
    sliced = formatted[start : start + limit]
    truncated = start + len(sliced) < len(formatted)

    output_parts = [
        f"<path>{display}</path>",
        "<type>directory</type>",
        "<entries>",
    ]
    output_parts.extend(sliced)

    if truncated:
        output_parts.append(
            f"\n(Showing {len(sliced)} of {len(formatted)} entries. "
            f"Use 'offset' parameter to read beyond entry {offset + len(sliced)})"
        )
    else:
        output_parts.append(f"\n({len(formatted)} entries)")
    output_parts.append("</entries>")
    text = "\n".join(output_parts)

    return ToolOutput(
        text,
        metadata={
            "display": {
                "type": "directory",
                "path": display,
                "entries": sliced,
                "offset": offset,
                "totalEntries": len(formatted),
                "truncated": truncated,
            },
        },
    )


def _format_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext != ".py":
        return False
    try:
        result = subprocess.run(
            ["ruff", "format", str(path)],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _write_safe_path(root: Path, raw_path: str) -> Path:
    path_str = raw_path.strip().strip("\"'")
    if not path_str:
        raise ValueError("path is required")
    filepath = resolve_absolute_path(root, path_str)
    if matches_blocked_pattern(filepath):
        raise ValueError(f"path is blocked: {_display(root, filepath)}")
    return filepath


def _write_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    request = _parse_write_file_request(root, operations, data)
    return with_file_mutation(
        request.path,
        lambda: _write_file_impl(root, request, operations, context_state),
    )


def _parse_write_file_request(
    root: Path,
    operations: FileOperations,
    data: ToolInput,
) -> WriteFileRequest:
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        raise ValueError("path is required")
    path = _write_safe_path(root, path_str)
    if operations.exists(path) and operations.is_dir(path):
        raise ValueError(f"path is a directory: {_display(root, path)}")
    if "content" not in data:
        raise ValueError("content is required")
    content = str(data.get("content", ""))
    _ensure_write_size(content)
    return WriteFileRequest(path=path, content=content)


def _write_file_impl(
    root: Path,
    request: WriteFileRequest,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
) -> str:
    operations.mkdir(request.path.parent)
    display = _display(root, request.path)

    old_content = ""
    desired_bom = False
    if operations.exists(request.path):
        raw = operations.read_bytes(request.path)
        desired_bom = raw.startswith(b"\xef\xbb\xbf")
        old_content = raw.decode("utf-8-sig" if desired_bom else "utf-8")

    new_has_bom = request.content.startswith("\ufeff")
    desired_bom = desired_bom or new_has_bom
    clean_content = request.content[1:] if new_has_bom else request.content

    diff = "".join(
        unified_diff(
            old_content.splitlines(keepends=True),
            clean_content.splitlines(keepends=True),
            fromfile=f"a/{display}",
            tofile=f"b/{display}",
        )
    )

    encoding = "utf-8-sig" if desired_bom else "utf-8"
    _write_text(request.path, clean_content, encoding, operations)
    _format_file(request.path)

    if context_state is not None:
        context_state.record_file(request.path)

    return ToolOutput(
        f"wrote file: {display}\n{diff}",
        metadata={"patch": diff},
    )


def _edit_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    request = _parse_edit_file_request(root, operations, data)
    return with_file_mutation(
        request.path,
        lambda: _edit_file_impl(root, request, operations, context_state),
    )


def _parse_edit_file_request(
    root: Path,
    operations: FileOperations,
    data: ToolInput,
) -> EditFileRequest:
    prepared = _prepare_edit_arguments(data)
    path_str = str(prepared.get("path", "")).strip()
    if not path_str:
        raise ValueError("path is required")
    path = _write_safe_path(root, path_str)
    if operations.exists(path) and operations.is_dir(path):
        raise ValueError(f"path is a directory: {_display(root, path)}")
    if not operations.is_file(path):
        raise ValueError(f"not a file: {_display(root, path)}")

    edits = _prepare_edits(prepared)
    if not edits:
        raise ValueError("no edits provided")

    return EditFileRequest(
        path=path,
        edits=edits,
        replace_all=bool(prepared.get("replace_all", False)),
    )


def _edit_file_impl(
    root: Path,
    request: EditFileRequest,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
) -> str:
    path = request.path
    content, encoding = _read_text(path, operations)
    bom, clean_content = strip_bom(content)
    ending = detect_line_ending(clean_content)
    normalized = normalize_to_lf(clean_content)
    display = _display(root, path)

    updated = normalized
    total_replacements = 0
    for edit in request.edits:
        try:
            result = apply_fuzzy_replace(
                updated,
                edit.old_text,
                edit.new_text,
                replace_all=request.replace_all and len(request.edits) == 1,
            )
        except ValueError as e:
            raise ValueError(f"{e} in {display}")
        if request.replace_all and len(request.edits) == 1:
            count = normalized.count(edit.old_text)
            total_replacements = count
        else:
            total_replacements += 1
        updated = result

    if updated == normalized:
        raise ValueError(
            f"No changes made to {display}. The replacement produced identical content."
        )

    restored = restore_line_endings(updated, ending)
    final = bom + restored
    _ensure_write_size(final)
    _write_text(path, final, encoding, operations)
    _format_file(path)

    if context_state is not None:
        context_state.record_file(path)

    diff = _diff_preview(display, content, final)
    return ToolOutput(
        f"edited file: {display} replacements={total_replacements}\n{diff}",
        metadata={
            "patch": diff,
            "first_changed_line": _first_changed_line(content, final),
        },
    )


def _prepare_edit_arguments(data: dict[str, Any]) -> dict[str, Any]:
    """归一化 LLM 可能输出的不规范 JSON 结构。

    LLM 有时会生成不符合 JSON schema 的 edit_file 输入，此函数处理已知偏差：

    1. edits 字段被序列化为字符串 → 尝试 JSON 解析并还原为数组
    2. old_text/new_text 被放在顶层而非 edits 数组中 → 自动合并到 edits 内

    移除条件：升级到足够可靠的模型版本后，可移除此归一化层，改用严格 schema 校验。
    """
    result = dict(data)

    raw_edits = result.get("edits")
    if isinstance(raw_edits, str):
        try:
            parsed = json.loads(raw_edits)
            if isinstance(parsed, list):
                result["edits"] = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    old_text = result.get("old_text")
    new_text = result.get("new_text")
    if old_text is not None and new_text is not None:
        edits = list(result.get("edits") or [])
        edits.append({"old_text": str(old_text), "new_text": str(new_text)})
        result["edits"] = edits
        result.pop("old_text", None)
        result.pop("new_text", None)

    return result


def read_project_text_file(project_root: Path, raw_path: str) -> str:
    operations = LocalFileOperations()
    root = project_root.resolve()
    path = _safe_path(root, raw_path)
    if not operations.is_file(path):
        raise ValueError(f"not a file: {_display(root, path)}")
    size = operations.size(path)
    if size > MAX_FILE_READ_BYTES:
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


def _prepare_edits(data: ToolInput) -> list[FileEdit]:
    edits: list[FileEdit] = []

    raw_edits = data.get("edits")
    if isinstance(raw_edits, list):
        for i, edit in enumerate(raw_edits):
            if not isinstance(edit, dict):
                raise ValueError(f"edits[{i}]: must be an object")
            old = str(edit.get("old_text", ""))
            new = str(edit.get("new_text", ""))
            if not old:
                raise ValueError(f"edits[{i}].old_text must not be empty")
            edits.append(FileEdit(old_text=old, new_text=new))

    old_text = str(data.get("old_text", "")).strip()
    if old_text:
        if "new_text" not in data:
            raise ValueError("new_text is required")
        edits.append(
            FileEdit(old_text=old_text, new_text=str(data.get("new_text", "")))
        )

    return edits


def _edit_payloads(edits: list[FileEdit]) -> list[dict[str, str]]:
    return [
        {
            "old_text": edit.old_text,
            "new_text": edit.new_text,
        }
        for edit in edits
    ]
