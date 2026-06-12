"""受沙箱约束的本地文件工具。

文件工具负责读写项目内文本文件，并在工具层集中处理路径解析、敏感目录
拒绝、输出截断和基于 old_text 的文件编辑（含模糊匹配）。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
import json
from pathlib import Path
from typing import Any, Protocol

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.skills import ToolInput, ToolOutput, resolve_project_path
from .edit_diff import (
    apply_edits_fuzzy,
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_image import _detect_image, _read_image
from .file_mutation_queue import with_file_mutation
from .path_utils import (
    is_path_blocked,
    resolve_read_path,
    truncate_output,
    display_path,
)
from .truncate import truncate_head, format_size

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000


@dataclass(frozen=True)
class ReadRange:
    offset: int = 1
    limit: int | None = None


@dataclass(frozen=True)
class ReadFileRequest:
    path: Path
    display_path: str
    read_range: ReadRange | None
    image_mime: str | None


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


def _read_file(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    data: ToolInput,
) -> str:
    request = _parse_read_file_request(root, operations, data)
    if request.image_mime:
        return _read_image(
            request.path,
            request.display_path,
            request.image_mime,
            operations,
        )

    if context_state is not None:
        context_state.record_file(request.path)
    text, _encoding = _read_text(request.path, operations)

    if request.read_range is not None:
        return _read_with_offset_limit(
            text,
            request.display_path,
            request.read_range,
        )

    return _read_full(text, request.display_path)


def _parse_read_file_request(
    root: Path,
    operations: FileOperations,
    data: ToolInput,
) -> ReadFileRequest:
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

    display = _display(root, path)
    mime = _detect_image(path, operations)
    return ReadFileRequest(
        path=path,
        display_path=display,
        read_range=_read_range(data),
        image_mime=mime,
    )


def _read_full(text: str, display_path: str) -> str:
    tr = truncate_head(text)
    if not tr.truncated:
        return tr.content

    if tr.first_line_exceeds_limit:
        return (
            f"[Line 1 is {format_size(tr.total_bytes)}, exceeds "
            f"{format_size(tr.max_bytes)} limit. "
            f"Use bash: sed -n '1p' {json.dumps(display_path)} | head -c {tr.max_bytes}]"
        )

    end_line = tr.output_lines
    continuation = json.dumps({"path": display_path, "offset": end_line + 1})
    reason = (
        "lines" if tr.truncated_by == "lines" else format_size(tr.max_bytes) + " limit"
    )
    return (
        f"{tr.content}\n\n"
        f"[Showing lines 1-{end_line} of {tr.total_lines} "
        f"({reason}). "
        f"Use read_file with {continuation} to continue.]"
    )


def _read_with_offset_limit(
    text: str,
    display_path: str,
    read_range: ReadRange,
) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    start = read_range.offset - 1
    if start >= len(lines):
        raise ValueError(
            f"offset {read_range.offset} is beyond end of file ({len(lines)} lines total)"
        )
    end = (
        len(lines)
        if read_range.limit is None
        else min(start + read_range.limit, len(lines))
    )
    selected = "\n".join(lines[start:end])
    selected_truncated = _truncate(selected)

    if end < len(lines):
        continuation: dict[str, int | str] = {
            "path": display_path,
            "offset": end + 1,
        }
        if read_range.limit is not None:
            continuation["limit"] = read_range.limit
        selected_truncated += (
            f"\n\n[Showing lines {read_range.offset}-{end} of {len(lines)}. "
            f"Use read_file with {json.dumps(continuation, ensure_ascii=False)} "
            "to continue.]"
        )
    return selected_truncated


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
    path = _safe_path(root, path_str)
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
    _write_text(request.path, request.content, "utf-8", operations)
    if context_state is not None:
        context_state.record_file(request.path)
    return f"wrote file: {_display(root, request.path)}"


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
    path = _safe_path(root, path_str)
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

    if request.replace_all and len(request.edits) == 1:
        edit = request.edits[0]
        count = normalized.count(edit.old_text)
        if count == 0:
            raise ValueError("old_text not found")
        updated = normalized.replace(edit.old_text, edit.new_text)
        edit_count = count
    else:
        updated, edit_count = apply_edits_fuzzy(
            normalized, _edit_payloads(request.edits), _display(root, path)
        )

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


def _read_range(data: ToolInput) -> ReadRange | None:
    offset = data.get("offset")
    limit = data.get("limit")
    if offset is None and limit is None:
        return None

    offset_value = 1 if offset is None else _parse_optional_int(offset, "offset")
    if offset_value < 1:
        raise ValueError("offset must be positive")

    limit_value = None if limit is None else _parse_optional_int(limit, "limit")
    if limit_value is not None and limit_value < 0:
        raise ValueError("limit must be non-negative")

    return ReadRange(offset=offset_value, limit=limit_value)


def _parse_optional_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


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
