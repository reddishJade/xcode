from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from xcode.harness.skills import ToolSpec

"""REPL 命令、工具名和 @file 引用补全。"""

if TYPE_CHECKING:
    from prompt_toolkit.completion import Completer
else:
    try:
        from prompt_toolkit.completion import Completer
    except ImportError:

        class Completer:
            pass


COMMANDS = (
    "/help",
    "/clear",
    "/rewind",
    "/resume",
    "/sessions",
    "/plan",
    "/review",
    "/act",
    "/compact",
    "/tool",
    "/exit",
)
MAX_FILE_COMPLETIONS = 100
BLOCKED_PARTS = {".git", ".venv", "__pycache__"}


@dataclass(frozen=True)
class CompletionItem:
    text: str
    start_position: int
    display_meta: str = ""


class ReplCompleter(Completer):
    def __init__(self, project_root: Path, registry: Iterable[ToolSpec] = ()) -> None:
        self.project_root = project_root.resolve()
        self.tool_names = tuple(sorted(tool.name for tool in registry))
        self._directory_cache: dict[Path, tuple[str, ...]] = {}

    def get_completions(self, document, _complete_event):
        try:
            from prompt_toolkit.completion import Completion
        except ImportError:
            return
        for item in self.complete(document.text_before_cursor):
            yield Completion(
                item.text,
                start_position=item.start_position,
                display_meta=item.display_meta,
            )

    async def get_completions_async(self, document, complete_event):
        for completion in self.get_completions(document, complete_event):
            yield completion

    def complete(self, text_before_cursor: str) -> list[CompletionItem]:
        if text_before_cursor.startswith("/tool "):
            return self._complete_tool_name(text_before_cursor)
        if text_before_cursor.startswith("/"):
            return self._complete_command(text_before_cursor)
        return self._complete_file_reference(text_before_cursor)

    def _complete_command(self, text: str) -> list[CompletionItem]:
        return [
            CompletionItem(command, -len(text), "command")
            for command in COMMANDS
            if command.startswith(text)
        ]

    def _complete_tool_name(self, text: str) -> list[CompletionItem]:
        parts = text.split(maxsplit=2)
        if len(parts) > 2:
            return []
        partial = parts[1] if len(parts) == 2 else ""
        return [
            CompletionItem(name, -len(partial), "tool")
            for name in self.tool_names
            if name.startswith(partial)
        ]

    def _complete_file_reference(self, text: str) -> list[CompletionItem]:
        marker = _active_file_marker(text)
        if marker is None:
            return []
        partial, start_position = marker
        return [
            CompletionItem(path, start_position, "file")
            for path in _matching_files(
                self.project_root, partial, self._directory_cache
            )
        ]


def _active_file_marker(text: str) -> tuple[str, int] | None:
    token_start = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t")) + 1
    token = text[token_start:]
    if not token.startswith("@"):
        return None
    partial = token[1:]
    if not partial:
        return None
    if any(char in partial for char in "*?[]"):
        return None
    return partial, -len(partial)


def _matching_files(
    root: Path,
    partial: str,
    directory_cache: dict[Path, tuple[str, ...]] | None = None,
) -> list[str]:
    directory_text, name_prefix = _split_partial(partial)
    try:
        directory = (root / directory_text).resolve()
        directory.relative_to(root)
    except ValueError:
        return []
    if not directory.is_dir() or _is_blocked(root, directory):
        return []

    matches = []
    if directory_cache is not None and directory in directory_cache:
        candidates = list(directory_cache[directory])
    else:
        candidates = _directory_entries(root, directory)
        if directory_cache is not None:
            directory_cache[directory] = tuple(candidates)

    for candidate in candidates:
        name = Path(candidate.rstrip("/")).name
        if name.startswith(name_prefix):
            matches.append(candidate)
        if len(matches) >= MAX_FILE_COMPLETIONS:
            break
    return matches


def _split_partial(partial: str) -> tuple[str, str]:
    normalized = partial.replace("\\", "/")
    if "/" not in normalized:
        return "", normalized
    directory, _, name_prefix = normalized.rpartition("/")
    return directory + "/", name_prefix


def _directory_entries(root: Path, directory: Path) -> list[str]:
    entries = []
    for path in sorted(
        directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
    ):
        if _is_blocked(root, path):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if path.is_dir():
            entries.append(relative + "/")
        elif path.is_file():
            entries.append(relative)
    return entries[:MAX_FILE_COMPLETIONS]


def _is_blocked(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True
    parts = set(relative.parts)
    if parts & BLOCKED_PARTS:
        return True
    if ".env" in relative.parts or relative.name == ".env":
        return True
    return (
        len(relative.parts) >= 3
        and relative.parts[0] == "xcode"
        and relative.parts[1] == ".local"
        and relative.parts[2] == "chroma_db"
    )
