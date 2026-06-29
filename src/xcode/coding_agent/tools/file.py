"""文件工具注册器与 Schema。"""

from __future__ import annotations

import threading
from pathlib import Path

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.skills import ToolSpec
from .apply_patch import build_apply_patch_tool
from .file_handlers import (
    _edit_file,
    _read_file,
    _write_file,
    FileOperations,
    LocalFileOperations,
)

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000

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
        build_apply_patch_tool(
            root,
            context_state=context_state,
            operations=ops,
        ),
    )
