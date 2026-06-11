from __future__ import annotations


from dataclasses import dataclass
from difflib import unified_diff
import json
import threading
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
from .truncate import truncate_head, format_size

"""受沙箱约束的本地文件工具。

文件工具负责读写项目内文本文件，并在工具层集中处理路径解析、敏感目录
拒绝、输出截断和基于 old_text 的文件编辑（含模糊匹配）。
"""

# 文件读写限制（防止内存耗尽和输出截断）
MAX_READ_BYTES = 1_000_000  # 单文件读取限制：1MB
MAX_WRITE_BYTES = 1_000_000  # 单文件写入限制：1MB


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
    cancel_event: threading.Event | None = None,
) -> tuple[ToolSpec, ...]:
    root = project_root.resolve()
    ops = operations or LocalFileOperations()

    def _handler(fn, data):
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Tool cancelled")
        return fn(data)

    return (
        ToolSpec(
            name="read_file",
            description="Read a text file inside the project sandbox.",
            input_hint='JSON: {"path": "src/xcode/main.py", "offset": 1, "limit": 80}',
            handler=lambda data: _handler(
                lambda d: _read_file(root, ops, context_state, d), data
            ),
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
            handler=lambda data: _handler(
                lambda d: _write_file(root, ops, context_state, d), data
            ),
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
            handler=lambda data: _handler(
                lambda d: _edit_file(root, ops, context_state, d), data
            ),
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


def _detect_image(path: Path, operations: FileOperations) -> str | None:
    try:
        buf = operations.read_bytes(path)
    except Exception:
        return None
    import filetype

    return filetype.guess_mime(buf)


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


def _read_image(
    path: Path, display_path: str, mime: str, operations: FileOperations
) -> str:
    import base64
    from io import BytesIO

    from PIL import Image

    data = operations.read_bytes(path)
    img = Image.open(BytesIO(data))
    orig_w, orig_h = img.width, img.height
    # 图片尺寸限制（Anthropic API 要求）
    max_dim = 2000
    if orig_w > max_dim or orig_h > max_dim:
        # 限制最大边为 2000px 以符合 Anthropic API 图片尺寸要求
        ratio = min(max_dim / orig_w, max_dim / orig_h)
        new_size = (int(orig_w * ratio), int(orig_h * ratio))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        save_format = img.format or "PNG"
        resized.save(buf, format=save_format)
        new_w, new_h = resized.width, resized.height
        data = buf.getvalue()
        _img_mime: dict[str, str] = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime = _img_mime.get(save_format.lower(), mime)
    else:
        new_w, new_h = orig_w, orig_h
    b64 = base64.b64encode(data).decode("ascii")
    hint = (
        f" (resized from {orig_w}x{orig_h} to {new_w}x{new_h})"
        if orig_w > max_dim or orig_h > max_dim
        else ""
    )
    return ToolOutput(
        f"Read image file [{mime}]{hint}\nImage data is available in metadata.",
        metadata={"image": {"mime": mime, "data": b64}},
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
    """对 edit_file 输入做防御性预处理，对齐 Pi-mono 的 prepareArguments。

    防御性设计原因：
    LLM 可能生成不规范的 JSON 结构，此函数归一化输入以提高容错性：
    - 字符串化数组 → 解析为真实数组
    - 扁平化字段 → 提取嵌套结构
    - 缺失必需字段 → 补充默认值

    避免因 LLM 输出格式偏差导致工具调用失败。
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
    """将输入归一化为 edits 列表。"""
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
