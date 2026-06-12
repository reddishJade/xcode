from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from xcode.harness.skills import ToolSpec

from .commands import COMMAND_GROUP_ORDER, CommandEntry
from .reasoning_effort import normalize_reasoning_effort_options

"""REPL 命令、工具名和 @file 引用补全。"""

if TYPE_CHECKING:
    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    from prompt_toolkit.completion import Completer
else:
    try:
        from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    except ImportError:

        class AutoSuggest:
            def get_suggestion(self, buffer: object, document: object) -> object:
                return None

        class Suggestion:
            def __init__(self, text: str) -> None:
                self.text = text

    try:
        from prompt_toolkit.completion import Completer
    except ImportError:

        class Completer:
            pass


MAX_FILE_COMPLETIONS = 100
BLOCKED_PARTS = {".git", ".venv", "__pycache__"}


class CommandArgsSuggester(AutoSuggest):
    """当输入栏恰好是已注册的命令时，以暗色提示显示参数用法。"""

    def __init__(self, command_args: dict[str, str]) -> None:
        self._command_args = command_args

    def get_suggestion(  # pyright: ignore[override]
        self, buffer: object, document: object
    ) -> Suggestion | None:
        text: str = getattr(document, "text", "")
        stripped = text.strip()
        if stripped in self._command_args:
            return Suggestion(f" {self._command_args[stripped]}")
        return None


@dataclass(frozen=True)
class CompletionItem:
    text: str
    start_position: int
    display_meta: str = ""


class ReplCompleter(Completer):
    def __init__(
        self,
        project_root: Path,
        registry: Iterable[ToolSpec] = (),
        command_names: Iterable[str] = (),
        command_registry: dict[str, CommandEntry] | None = None,
        effort_options: Iterable[str] | Callable[[], Iterable[str]] = (),
        model_options: Iterable[str] | Callable[[], Iterable[str]] = (),
    ) -> None:
        self.project_root = project_root.resolve()
        self.tool_names = tuple(sorted(tool.name for tool in registry))
        self.command_names = tuple(command_names)
        self._command_meta: dict[str, str] = {}
        self._command_args: dict[str, str] = {}
        self._command_group: dict[str, str] = {}
        if command_registry:
            for name, entry in command_registry.items():
                if entry.visible:
                    self._command_meta[name] = entry.desc
                    self._command_group[name] = entry.group
                    if entry.args_desc:
                        self._command_args[name] = entry.args_desc
        self._effort_options = effort_options
        self._model_options = model_options
        self._directory_cache: dict[Path, tuple[str, ...]] = {}

    @property
    def command_args(self) -> dict[str, str]:
        return self._command_args

    def get_completions(self, document, complete_event):
        try:
            from prompt_toolkit.completion import Completion
            from prompt_toolkit.formatted_text import FormattedText
        except ImportError:
            return
        for item in self.complete(document.text_before_cursor):
            item_display_meta: str = item.display_meta
            meta: object = (
                FormattedText([("fg:ansibrightblack", item_display_meta)])
                if item_display_meta else None
            )
            yield Completion(
                item.text,
                start_position=item.start_position,
                display_meta=meta,
            )

    async def get_completions_async(self, document, complete_event):
        for completion in self.get_completions(document, complete_event):
            yield completion

    def complete(self, text_before_cursor: str) -> list[CompletionItem]:
        if (
            text_before_cursor.startswith("/effort")
            and len(text_before_cursor) > len("/effort")
            and text_before_cursor[len("/effort")].isspace()
        ):
            return self._complete_effort(text_before_cursor)
        if (
            text_before_cursor.startswith("/model")
            and len(text_before_cursor) > len("/model")
            and text_before_cursor[len("/model")].isspace()
        ):
            return self._complete_model(text_before_cursor)
        if text_before_cursor.startswith("/tool "):
            return self._complete_tool_name(text_before_cursor)
        if text_before_cursor.startswith("/"):
            return self._complete_command(text_before_cursor)
        if text_before_cursor.startswith("!"):
            return self._complete_shell(text_before_cursor)
        return self._complete_file_reference(text_before_cursor)

    def _complete_command(self, text: str) -> list[CompletionItem]:
        matched = [
            command for command in self.command_names if command.startswith(text)
        ]
        matched.sort(
            key=lambda name: (
                COMMAND_GROUP_ORDER.get(self._command_group.get(name, ""), 99),
                name,
            )
        )
        return [
            CompletionItem(
                command,
                -len(text),
                display_meta=self._command_meta.get(command, ""),
            )
            for command in matched
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

    def _complete_effort(self, text: str) -> list[CompletionItem]:
        parts = text.split(maxsplit=1)
        partial = parts[1] if len(parts) == 2 else ""
        options = self._current_effort_options()
        if not partial:
            return [
                CompletionItem(option, -len(partial), "effort") for option in options
            ]
        return [
            CompletionItem(option, -len(partial), "effort")
            for option in options
            if option.startswith(partial)
        ]

    def _complete_model(self, text: str) -> list[CompletionItem]:
        parts = text.split(maxsplit=2)
        if len(parts) > 2:
            return []
        partial = parts[1] if len(parts) == 2 else ""
        options = self._current_model_options()
        if not partial:
            return [
                CompletionItem(option, -len(partial), "model") for option in options
            ]
        return [
            CompletionItem(option, -len(partial), "model")
            for option in options
            if option.startswith(partial)
        ]

    def _current_model_options(self) -> tuple[str, ...]:
        options = (
            self._model_options()
            if callable(self._model_options)
            else self._model_options
        )
        return tuple(options)

    def _current_effort_options(self) -> tuple[str, ...]:
        options = (
            self._effort_options()
            if callable(self._effort_options)
            else self._effort_options
        )
        return normalize_reasoning_effort_options(options)

    def _complete_shell(self, text: str) -> list[CompletionItem]:
        marker = _active_shell_word(text)
        if marker is None:
            return []
        word, start_position, word_index = marker
        if word_index == 0:
            return []

        partial = _unescape_shell_word(word)
        if any(char in partial for char in "*?[]$"):
            return []
        return [
            CompletionItem(_escape_shell_path(path), start_position, "file")
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


def _active_shell_word(text: str) -> tuple[str, int, int] | None:
    shell_text = text[1:]
    words = _shell_words(shell_text)
    if not words:
        return "", 0, 0
    last = words[-1]
    if shell_text and shell_text[-1].isspace():
        return "", 0, len(words)
    return last[0], -(len(shell_text) - last[1]), len(words) - 1


def _shell_words(text: str) -> list[tuple[str, int]]:
    words: list[tuple[str, int]] = []
    current: list[str] = []
    start: int | None = None
    quote: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            if start is None:
                start = index - 1
            current.append("\\" + char)
            escaped = False
            continue
        if char == "\\":
            if start is None:
                start = index
            escaped = True
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            if start is None:
                start = index
            quote = char
            current.append(char)
            continue
        if char.isspace():
            if start is not None:
                words.append(("".join(current), start))
                current = []
                start = None
            continue
        if start is None:
            start = index
        current.append(char)

    if escaped:
        current.append("\\")
    if start is not None:
        words.append(("".join(current), start))
    return words


def _unescape_shell_word(word: str) -> str:
    if len(word) >= 1 and word[0] in {"'", '"'}:
        word = word[1:]
    if len(word) >= 1 and word[-1] in {"'", '"'}:
        word = word[:-1]

    result = []
    escaped = False
    for char in word:
        if escaped:
            result.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            result.append(char)
    if escaped:
        result.append("\\")
    return "".join(result)


def _escape_shell_path(path: str) -> str:
    return "".join(
        "\\" + char if char in " \t\n'\"\\$`!&;()<>|" else char for char in path
    )


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
