"""read_file 工具注册器与 Schema。"""

from __future__ import annotations

import threading
from pathlib import Path

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.skills import ToolSpec

from .file_handlers import FileOperations, LocalFileOperations, _read_file

READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The absolute path to the file or directory to read.",
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


def build_read_file_tool(
    project_root: Path,
    context_state: ContextualRetrievalState | None = None,
    operations: FileOperations | None = None,
    cancel_event: threading.Event | None = None,
) -> ToolSpec:
    root = project_root.resolve()
    ops = operations or LocalFileOperations()

    def handler(data):
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Tool cancelled")
        return _read_file(root, ops, context_state, data)

    return ToolSpec(
        name="read_file",
        description="Read a file or directory from the local filesystem.",
        input_hint='JSON: {"path": "/absolute/path/to/file", "offset": 1, "limit": 80}',
        handler=handler,
        schema=READ_FILE_SCHEMA,
        prompt_snippet="Read a text file inside the project sandbox",
        prompt_guidelines=(
            "Use read_file offset and limit to continue reading long files.",
        ),
    )
