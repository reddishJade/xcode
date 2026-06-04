from __future__ import annotations


from difflib import unified_diff
from pathlib import Path
from typing import Any

from ..agent_runtime.contextual import ContextualRetrievalState
from ..skills import ToolInput, ToolSpec, resolve_project_path
from .path_utils import is_path_blocked, truncate_output, display_path

"""受沙箱约束的本地文件工具。

文件工具负责读写项目内文本文件，并在工具层集中处理路径解析、敏感目录
拒绝、输出截断和基于 old_text 精确匹配的文件编辑。
"""

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000


def build_file_tools(
    project_root: Path,
    context_state: ContextualRetrievalState | None = None,
) -> tuple[ToolSpec, ...]:
    root = project_root.resolve()

    def read_file(data: ToolInput) -> str:
        path_str = str(data.get("path", "")).strip()
        if not path_str:
            raise ValueError("path is required")
        path = _safe_path(root, path_str)
        if not path.is_file():
            raise ValueError(f"not a file: {_display(root, path)}")
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"file too large: {size} bytes")
        text, _encoding = _read_text(path)
        if context_state is not None:
            context_state.record_file(path)
        limit = data.get("limit")
        if limit is not None:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError) as exc:
                raise ValueError("limit must be an integer") from exc
            if limit_value < 0:
                raise ValueError("limit must be non-negative")
            lines = text.splitlines()
            text = "\n".join(lines[:limit_value])
        return _truncate(text)

    def write_file(data: ToolInput) -> str:
        path_str = str(data.get("path", "")).strip()
        if not path_str:
            raise ValueError("path is required")
        path = _safe_path(root, path_str)
        if path.exists() and path.is_dir():
            raise ValueError(f"path is a directory: {_display(root, path)}")
        if "content" not in data:
            raise ValueError("content is required")
        content = str(data.get("content", ""))
        _ensure_write_size(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, content, "utf-8")
        if context_state is not None:
            context_state.record_file(path)
        return f"wrote file: {_display(root, path)}"

    def edit_file(data: ToolInput) -> str:
        path_str = str(data.get("path", "")).strip()
        if not path_str:
            raise ValueError("path is required")
        path = _safe_path(root, path_str)
        if path.exists() and path.is_dir():
            raise ValueError(f"path is a directory: {_display(root, path)}")
        if not path.is_file():
            raise ValueError(f"not a file: {_display(root, path)}")

        edits = _prepare_edits(data)
        if not edits:
            raise ValueError("no edits provided")

        content, encoding = _read_text(path)
        updated, edit_count = _apply_edits(
            content,
            edits,
            _display(root, path),
            replace_all=bool(data.get("replace_all", False)),
        )
        _ensure_write_size(updated)
        _write_text(path, updated, encoding)
        if context_state is not None:
            context_state.record_file(path)
        diff = _diff_preview(_display(root, path), content, updated)
        return f"edited file: {_display(root, path)} replacements={edit_count}\n{diff}"

    return (
        ToolSpec(
            name="read_file",
            description="Read a text file inside the project sandbox.",
            input_hint='JSON: {"path": "src/xcode/main.py", "limit": 80}',
            handler=read_file,
            risk="low",
            schema={
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
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            read_only=True,
            concurrency_safe=True,
            group="core",
        ),
        ToolSpec(
            name="write_file",
            description="Write a text file inside the project sandbox.",
            input_hint='JSON: {"path": "notes.md", "content": "..."}',
            handler=write_file,
            risk="high",
            schema={
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
            },
            group="core",
            counts_as_progress=True,
        ),
        ToolSpec(
            name="edit_file",
            description="Edit a text file with one or more search-and-replace edits. Each edit's old_text is matched against the original file content (not incrementally). Use this tool for targeted replacements.",
            input_hint='JSON: {"path": "src/main.py", "edits": [{"old_text": "...", "new_text": "..."}]}',
            handler=edit_file,
            risk="high",
            schema={
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
            },
            group="core",
            counts_as_progress=True,
        ),
    )


def read_project_text_file(project_root: Path, raw_path: str) -> str:
    root = project_root.resolve()
    path = _safe_path(root, raw_path)
    if not path.is_file():
        raise ValueError(f"not a file: {_display(root, path)}")
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ValueError(f"file too large: {size} bytes")
    text, _encoding = _read_text(path)
    return _truncate(text)


def _safe_path(root: Path, raw_path: str) -> Path:
    path = resolve_project_path(root, raw_path)
    if is_path_blocked(root, path):
        raise ValueError(f"path is blocked: {display_path(root, path)}")
    return path


def _is_blocked(root: Path, path: Path) -> bool:
    return is_path_blocked(root, path)


def _truncate(text: str) -> str:
    return truncate_output(text)


def _read_text(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        return data.decode(encoding), encoding
    except UnicodeDecodeError:
        pass
    raise UnicodeDecodeError("utf-8", data, 0, 1, "file is not valid UTF-8 text")


def _write_text(path: Path, text: str, encoding: str) -> None:
    path.write_bytes(text.encode(encoding))


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


def _apply_edits(
    content: str,
    edits: list[dict[str, str]],
    display_path: str,
    replace_all: bool = False,
) -> tuple[str, int]:
    """对原始内容应用编辑。所有编辑基于原始内容匹配，逆序替换。

    返回 (更新后内容, 替换次数) 或抛出 ValueError。
    """
    if replace_all and len(edits) == 1:
        edit = edits[0]
        count = content.count(edit["old_text"])
        if count == 0:
            raise ValueError("old_text not found")
        updated = content.replace(edit["old_text"], edit["new_text"])
        return updated, count

    matches: list[dict[str, Any]] = []
    for i, edit in enumerate(edits):
        old = edit["old_text"]
        if not old:
            raise ValueError(f"edits[{i}].old_text must not be empty in {display_path}")

        idx = content.find(old)
        if idx == -1:
            raise ValueError(
                f"Could not find edits[{i}] in {display_path}. "
                "The old text must match exactly."
            )
        idx2 = content.find(old, idx + 1)
        if idx2 != -1:
            raise ValueError(
                f"Found multiple occurrences of edits[{i}] in {display_path}. "
                "Each old_text must be unique. Use more context to disambiguate."
            )
        matches.append(
            {
                "index": i,
                "start": idx,
                "end": idx + len(old),
                "old": old,
                "new": edit["new_text"],
            }
        )

    # 检测重叠
    matches.sort(key=lambda m: m["start"])
    for a, b in zip(matches, matches[1:]):
        if a["end"] > b["start"]:
            raise ValueError(
                f"edits[{a['index']}] and edits[{b['index']}] overlap in {display_path}. "
                "Merge them into one edit or target disjoint regions."
            )

    # 逆序替换
    new_content = content
    for m in reversed(matches):
        new_content = new_content[: m["start"]] + m["new"] + new_content[m["end"] :]

    if new_content == content:
        raise ValueError(
            f"No changes made to {display_path}. The replacements produced identical content."
        )

    return new_content, len(matches)
