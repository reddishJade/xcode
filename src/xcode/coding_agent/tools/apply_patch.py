"""apply_patch 工具实现。

该模块实现结构化 patch 语法，支持新增、更新、删除和移动文件。
工具执行前会完整解析 patch 并校验所有目标路径，
执行时对原文做精确匹配，避免静默覆盖并发修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Literal, cast

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.skills import ToolInput, ToolOutput, ToolSpec

from .edit_diff import (
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_handlers import (
    FileOperations,
    LocalFileOperations,
    _ensure_write_size,
    _read_text,
    _format_file,
)
from .file_mutation_queue import with_file_mutation
from .path_utils import (
    display_path,
    resolve_absolute_path,
    matches_blocked_pattern,
)

PatchKind = Literal["add", "update", "delete", "move"]
LineOp = Literal[" ", "+", "-"]


APPLY_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "patch_text": {
            "type": "string",
            "description": "Full patch text beginning with *** Begin Patch and ending with *** End Patch.",
        },
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PatchLine:
    op: LineOp
    text: str


@dataclass(frozen=True)
class PatchSection:
    anchor: str
    lines: tuple[PatchLine, ...]


@dataclass(frozen=True)
class PatchHunk:
    kind: PatchKind
    path: str
    sections: tuple[PatchSection, ...] = ()
    add_lines: tuple[str, ...] = ()
    move_path: str | None = None


@dataclass(frozen=True)
class FileChange:
    kind: PatchKind
    path: Path
    display_path: str
    before: str
    after: str
    move_path: Path | None = None
    move_display_path: str | None = None


def build_apply_patch_tool(
    project_root: Path,
    context_state: ContextualRetrievalState | None = None,
    operations: FileOperations | None = None,
) -> ToolSpec:
    """构建 apply_patch 工具。"""
    root = project_root.resolve()
    ops = operations or LocalFileOperations()

    def apply_patch(data: ToolInput) -> str:
        patch_text = _patch_text(data)
        hunks = parse_patch(patch_text)
        changes = _plan_changes(root, ops, hunks)
        return _apply_changes(root, ops, context_state, changes)

    return ToolSpec(
        name="apply_patch",
        description=(
            "Apply a structured patch inside the project sandbox. Supports "
            "Add File, Update File, Delete File, and Move to hunks."
        ),
        input_hint='JSON: {"patch_text": "*** Begin Patch\\n*** Update File: /abs/path/to/app.py\\n@@\\n-old\\n+new\\n*** End Patch"}',
        handler=apply_patch,
        schema=APPLY_PATCH_SCHEMA,
        group="core",
        counts_as_progress=True,
        prompt_snippet="Apply structured file patches with add, update, delete, and move hunks",
        prompt_guidelines=(
            "Use apply_patch for multi-file edits when exact old_text replacements are awkward.",
            "Each apply_patch hunk must start with *** Begin Patch and end with *** End Patch.",
            "Patch paths must be project-relative and may not escape the project sandbox.",
        ),
    )


def parse_patch(patch_text: str) -> tuple[PatchHunk, ...]:
    """解析完整 patch 文本。"""
    text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != "*** Begin Patch":
        raise ValueError("apply_patch verification failed: missing *** Begin Patch")
    if lines[-1] != "*** End Patch":
        raise ValueError("apply_patch verification failed: missing *** End Patch")
    if len(lines) == 2:
        raise ValueError("patch rejected: empty patch")

    hunks: list[PatchHunk] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            hunk, index = _parse_add_hunk(lines, index)
        elif line.startswith("*** Update File: "):
            hunk, index = _parse_update_hunk(lines, index)
        elif line.startswith("*** Delete File: "):
            hunk, index = _parse_delete_hunk(lines, index)
        else:
            raise ValueError(
                f"apply_patch verification failed: unexpected line {index + 1}: {line}"
            )
        hunks.append(hunk)

    if not hunks:
        raise ValueError("apply_patch verification failed: no hunks found")
    return tuple(hunks)


def extract_patch_paths(tool_input: object) -> tuple[str, ...]:
    """从 apply_patch 输入中提取写入目标路径，供权限模型使用。"""
    if not isinstance(tool_input, dict):
        return ()
    raw_paths = tool_input.get("paths")
    paths: list[str] = []
    if isinstance(raw_paths, list | tuple):
        paths.extend(path for path in raw_paths if isinstance(path, str))
    raw_path = tool_input.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        paths.append(raw_path)

    raw_patch = tool_input.get("patch_text")
    if raw_patch is None:
        raw_patch = tool_input.get("patchText")
    if isinstance(raw_patch, str) and raw_patch.strip():
        try:
            for hunk in parse_patch(raw_patch):
                paths.append(hunk.path)
                if hunk.move_path is not None:
                    paths.append(hunk.move_path)
        except ValueError:
            pass

    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


def _parse_add_hunk(lines: list[str], index: int) -> tuple[PatchHunk, int]:
    path = _header_path(lines[index], "*** Add File: ")
    index += 1
    add_lines: list[str] = []
    while index < len(lines) - 1 and not lines[index].startswith("*** "):
        line = lines[index]
        if not line.startswith("+"):
            raise ValueError(
                f"apply_patch verification failed: Add File line {index + 1} must start with +"
            )
        add_lines.append(line[1:])
        index += 1
    if not add_lines:
        raise ValueError("apply_patch verification failed: Add File requires content")
    return PatchHunk(kind="add", path=path, add_lines=tuple(add_lines)), index


def _parse_update_hunk(lines: list[str], index: int) -> tuple[PatchHunk, int]:
    path = _header_path(lines[index], "*** Update File: ")
    index += 1
    move_path: str | None = None
    if index < len(lines) - 1 and lines[index].startswith("*** Move to: "):
        move_path = _header_path(lines[index], "*** Move to: ")
        index += 1

    sections: list[PatchSection] = []
    current_anchor = ""
    current_lines: list[PatchLine] = []
    saw_change = False

    while index < len(lines) - 1:
        line = lines[index]
        if line == "*** End of File":
            index += 1
            break
        if line.startswith("*** "):
            break
        if line.startswith("@@"):
            if current_lines:
                sections.append(PatchSection(current_anchor, tuple(current_lines)))
                current_lines = []
            current_anchor = line[2:].strip()
            index += 1
            continue
        if not line:
            raise ValueError(
                f"apply_patch verification failed: update line {index + 1} is missing an operation prefix"
            )
        op = line[0]
        if op not in {" ", "+", "-"}:
            raise ValueError(
                f"apply_patch verification failed: invalid update operation {op!r} on line {index + 1}"
            )
        if op in {"+", "-"}:
            saw_change = True
        current_lines.append(PatchLine(op=cast(LineOp, op), text=line[1:]))
        index += 1

    if current_lines:
        sections.append(PatchSection(current_anchor, tuple(current_lines)))
    if not sections and move_path is None:
        raise ValueError(
            "apply_patch verification failed: Update File requires changes"
        )
    if not saw_change and move_path is None:
        raise ValueError("apply_patch verification failed: Update File has no edits")
    return (
        PatchHunk(
            kind="move" if move_path is not None else "update",
            path=path,
            sections=tuple(sections),
            move_path=move_path,
        ),
        index,
    )


def _parse_delete_hunk(lines: list[str], index: int) -> tuple[PatchHunk, int]:
    path = _header_path(lines[index], "*** Delete File: ")
    index += 1
    return PatchHunk(kind="delete", path=path), index


def _header_path(line: str, prefix: str) -> str:
    path = line.removeprefix(prefix).strip()
    if not path:
        raise ValueError(f"apply_patch verification failed: empty path in {prefix}")
    return path


def _patch_text(data: ToolInput) -> str:
    raw = data.get("patch_text")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("patch_text is required")
    return raw


def _plan_changes(
    root: Path,
    operations: FileOperations,
    hunks: tuple[PatchHunk, ...],
) -> tuple[FileChange, ...]:
    changes: list[FileChange] = []
    for hunk in hunks:
        path = _safe_path(root, hunk.path)
        display = display_path(root, path)
        if hunk.kind == "add":
            changes.append(_plan_add(root, operations, hunk, path, display))
        elif hunk.kind == "delete":
            changes.append(_plan_delete(root, operations, path, display))
        else:
            changes.append(_plan_update(root, operations, hunk, path, display))
    return tuple(changes)


def _plan_add(
    root: Path,
    operations: FileOperations,
    hunk: PatchHunk,
    path: Path,
    display: str,
) -> FileChange:
    if operations.exists(path):
        raise ValueError(
            f"apply_patch verification failed: file already exists: {display}"
        )
    after = "\n".join(hunk.add_lines)
    if after and not after.endswith("\n"):
        after += "\n"
    _ensure_write_size(after)
    return FileChange(
        kind="add", path=path, display_path=display, before="", after=after
    )


def _plan_delete(
    root: Path,
    operations: FileOperations,
    path: Path,
    display: str,
) -> FileChange:
    before, _encoding = _existing_text(root, operations, path, display)
    return FileChange(
        kind="delete",
        path=path,
        display_path=display,
        before=before,
        after="",
    )


def _plan_update(
    root: Path,
    operations: FileOperations,
    hunk: PatchHunk,
    path: Path,
    display: str,
) -> FileChange:
    before, _encoding = _existing_text(root, operations, path, display)
    bom, clean_before = strip_bom(before)
    ending = detect_line_ending(clean_before)
    normalized = normalize_to_lf(clean_before)
    updated = _apply_sections(normalized, hunk.sections, display)
    after = bom + restore_line_endings(updated, ending)
    _ensure_write_size(after)

    move_path = None
    move_display = None
    kind: PatchKind = "update"
    if hunk.move_path is not None:
        move_path = _safe_path(root, hunk.move_path)
        move_display = display_path(root, move_path)
        if move_path != path and operations.exists(move_path):
            raise ValueError(
                f"apply_patch verification failed: move target already exists: {move_display}"
            )
        kind = "move"
    return FileChange(
        kind=kind,
        path=path,
        display_path=display,
        before=before,
        after=after,
        move_path=move_path,
        move_display_path=move_display,
    )


def _apply_sections(
    content: str,
    sections: tuple[PatchSection, ...],
    display_path: str,
) -> str:
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
        trailing_newline = True
    else:
        trailing_newline = False

    cursor = 0
    for section in sections:
        old_lines = tuple(line.text for line in section.lines if line.op != "+")
        new_lines = tuple(line.text for line in section.lines if line.op != "-")
        if not old_lines:
            insert_at = _find_anchor(lines, section.anchor, cursor)
            lines[insert_at:insert_at] = list(new_lines)
            cursor = insert_at + len(new_lines)
            trailing_newline = True
            continue

        match = _find_sequence(lines, old_lines, cursor)
        if match is None and cursor:
            match = _find_sequence(lines, old_lines, 0)
        if match is None:
            raise ValueError(
                f"apply_patch verification failed: context not found in {display_path}"
            )
        lines[match : match + len(old_lines)] = list(new_lines)
        cursor = match + len(new_lines)
        trailing_newline = True

    result = "\n".join(lines)
    if trailing_newline:
        result += "\n"
    return result


def _find_sequence(
    lines: list[str],
    needle: tuple[str, ...],
    start: int,
) -> int | None:
    if not needle:
        return start
    limit = len(lines) - len(needle) + 1
    for index in range(max(start, 0), max(limit, 0)):
        if tuple(lines[index : index + len(needle)]) == needle:
            return index
    return None


def _find_anchor(lines: list[str], anchor: str, cursor: int) -> int:
    if not anchor:
        return cursor
    for index in range(max(cursor, 0), len(lines)):
        if anchor in lines[index]:
            return index + 1
    for index, line in enumerate(lines):
        if anchor in line:
            return index + 1
    return cursor


def _apply_changes(
    root: Path,
    operations: FileOperations,
    context_state: ContextualRetrievalState | None,
    changes: tuple[FileChange, ...],
) -> ToolOutput:
    if not changes:
        raise ValueError("apply_patch verification failed: no changes")

    def mutate() -> ToolOutput:
        for change in changes:
            _write_change(operations, change)
            target = change.move_path or change.path
            if target and change.kind != "delete":
                _format_file(target)
            if context_state is not None and change.kind != "delete":
                context_state.record_file(target)
        summary = "\n".join(_summary_line(change) for change in changes)
        diff = "\n".join(_diff_for_change(change) for change in changes)
        return ToolOutput(
            f"Success. Updated the following files:\n{summary}",
            metadata={
                "patch": diff,
                "files": [_file_metadata(change) for change in changes],
            },
        )

    return with_file_mutation(root, mutate)


def _write_change(operations: FileOperations, change: FileChange) -> None:
    target = change.move_path or change.path
    if change.kind == "delete":
        _remove_path(operations, change.path)
        return
    operations.mkdir(target.parent)
    operations.write_bytes(target, _encode_like(change.before, change.after))
    if (
        change.kind == "move"
        and change.move_path is not None
        and change.move_path != change.path
    ):
        _remove_path(operations, change.path)


def _encode_like(before: str, after: str) -> bytes:
    encoding = "utf-8-sig" if before.startswith("\ufeff") else "utf-8"
    return after.encode(encoding)


def _remove_path(operations: FileOperations, path: Path) -> None:
    operations.remove_file(path)


def _existing_text(
    root: Path,
    operations: FileOperations,
    path: Path,
    display: str,
) -> tuple[str, str]:
    if not operations.exists(path) or not operations.is_file(path):
        raise ValueError(
            f"apply_patch verification failed: failed to read file: {display}"
        )
    if matches_blocked_pattern(path):
        raise ValueError(f"path is blocked: {display}")
    return _read_text(path, operations)


def _safe_path(root: Path, raw_path: str) -> Path:
    path = resolve_absolute_path(root, raw_path)
    if matches_blocked_pattern(path):
        raise ValueError(f"path is blocked: {display_path(root, path)}")
    return path


def _summary_line(change: FileChange) -> str:
    if change.kind == "add":
        return f"A {change.display_path}"
    if change.kind == "delete":
        return f"D {change.display_path}"
    if change.kind == "move":
        return f"R {change.display_path} -> {change.move_display_path}"
    return f"M {change.display_path}"


def _diff_for_change(change: FileChange) -> str:
    from_path = change.display_path
    to_path = change.move_display_path or change.display_path
    return "".join(
        unified_diff(
            change.before.splitlines(keepends=True),
            change.after.splitlines(keepends=True),
            fromfile=f"a/{from_path}",
            tofile=f"b/{to_path}",
        )
    )


def _file_metadata(change: FileChange) -> dict[str, object]:
    additions, deletions = _change_counts(change.before, change.after)
    return {
        "path": change.display_path,
        "target_path": change.move_display_path or change.display_path,
        "type": change.kind,
        "patch": _diff_for_change(change),
        "additions": additions,
        "deletions": deletions,
    }


def _change_counts(before: str, after: str) -> tuple[int, int]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    additions = 0
    deletions = 0
    matcher = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            deletions += before_end - before_start
        if tag in {"replace", "insert"}:
            additions += after_end - after_start
    return additions, deletions
