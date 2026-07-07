"""write_file 与 edit_file 工具注册器与 Schema。"""

from __future__ import annotations

import threading
from pathlib import Path

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.agent.types import ToolSpec

from .file_handlers import FileOperations, LocalFileOperations, _edit_file, _write_file

WRITE_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The absolute path to the file to write (must be absolute, not relative).",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file.",
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
            "description": "The absolute path to the file to modify.",
        },
        "old_text": {
            "type": "string",
            "description": "The text to replace (must not be empty).",
        },
        "new_text": {
            "type": "string",
            "description": "The text to replace it with (must be different from old_text).",
        },
        "replace_all": {
            "type": "boolean",
            "description": "Replace all occurrences of old_text (default false).",
        },
    },
    "required": ["path", "old_text", "new_text"],
    "additionalProperties": False,
}


def build_write_file_tools(
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
            name="write_file",
            description=(
                "Create a new text file or intentionally replace an entire file. "
                "Prefer edit_file for targeted changes to an existing file."
            ),
            input_hint='JSON: {"path": "/absolute/path/to/file", "content": "..."}',
            handler=lambda data, _on_update=None: _handler(
                lambda d: _write_file(root, ops, context_state, d), data
            ),
            schema=WRITE_FILE_SCHEMA,
            prompt_snippet="Create new files or deliberately replace entire files",
            prompt_guidelines=(
                "Use write_file only for new files or deliberate full-file rewrites.",
            ),
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Modify an existing text file with a targeted replacement. "
                "old_text must match exactly, including whitespace and newlines. "
                "Use write_file for new files or full replacements."
            ),
            input_hint='JSON: {"path": "/absolute/path/to/file", "old_text": "...", "new_text": "..."}',
            handler=lambda data, _on_update=None: _handler(
                lambda d: _edit_file(root, ops, context_state, d), data
            ),
            schema=EDIT_FILE_SCHEMA,
            prompt_snippet=(
                "Make precise file edits with exact old_text/new_text replacements"
            ),
            prompt_guidelines=(
                "Use edit_file for precise changes to existing files.",
                "When editing, provide the full old_text that should be replaced.",
                "Keep old_text as small as possible while still unique in the file.",
            ),
        ),
    )
