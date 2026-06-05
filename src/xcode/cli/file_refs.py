from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

from xcode.coding_agent.tools.file import read_project_text_file

"""REPL 输入中的显式 @file 引用解析。"""

FILE_REF_PATTERN = re.compile(r"(?<!\S)@([^\s]+)")


@dataclass(frozen=True)
class FileReference:
    path: str
    status: str
    content: str = ""
    error: str = ""


def expand_file_references(
    text: str, project_root: Path
) -> tuple[str, list[FileReference]]:
    references: list[FileReference] = []
    for match in FILE_REF_PATTERN.finditer(text):
        raw_path = match.group(1).rstrip(".,;:)")
        if not raw_path:
            continue
        try:
            content = read_project_text_file(project_root, raw_path)
            references.append(
                FileReference(path=raw_path, status="ok", content=content)
            )
        except Exception as exc:
            references.append(
                FileReference(path=raw_path, status="error", error=str(exc))
            )
    if not references:
        return text, []

    blocks = []
    for reference in references:
        if reference.status == "ok":
            blocks.append(
                f'<file-reference path="{escape(reference.path, quote=True)}">\n'
                f"{reference.content}\n"
                "</file-reference>"
            )
        else:
            blocks.append(
                f'<file-reference path="{escape(reference.path, quote=True)}" '
                f'error="{escape(reference.error, quote=True)}" />'
            )
    expanded = "\n\n".join(blocks) + "\n\n<user-message>\n" + text + "\n</user-message>"
    return expanded, references
