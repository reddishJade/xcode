from __future__ import annotations

from pathlib import Path
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterable
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .commands import CommandEntry, PromptLike, PromptText
from .completion import CommandArgsSuggester, ReplCompleter
from xcode.harness.skills import ToolSpec


_MIN_REASONING_SUMMARY_SECONDS = 0.5
_MIN_REASONING_SUMMARY_CHARS = 24

CLI_COLOR_TITLE = "bold"
CLI_COLOR_DIM = "grey50"
CLI_COLOR_USER = "green bold"
CLI_COLOR_ASSISTANT = "cyan"
CLI_COLOR_THINKING = "grey50"
CLI_COLOR_TOOL = "yellow"
CLI_COLOR_SUCCESS = "green"
CLI_COLOR_ERROR = "red"
CLI_COLOR_WARNING = "yellow"
CLI_COLOR_INFO = "blue"
CLI_PROMPT_MARKER_STYLE = "ansigreen bold"
REPL_PROMPT_STYLE = {
    "prompt-marker": CLI_PROMPT_MARKER_STYLE,
    "suggestion": "fg:ansibrightblack",
    "completion-menu": "bg:default fg:default",
    "completion-menu.completion": "bg:default fg:default",
    "completion-menu.completion.current": "bg:default fg:default bold underline",
    "completion-menu.meta.completion": "bg:default fg:ansibrightblack",
    "completion-menu.meta.completion.current": (
        "bg:default fg:ansibrightblack bold underline"
    ),
    "completion-menu.multi-column-meta": "bg:default fg:ansibrightblack",
    "scrollbar.background": "bg:default",
    "scrollbar.button": "bg:default",
}


class PromptSessionAdapter:
    def __init__(self, session: Any) -> None:
        self.session = session

    def prompt(self, prompt_text: PromptText) -> str:
        return str(self.session.prompt(prompt_text))


def reasoning_preview_lines(text: str, width: int | None = None) -> list[str]:
    width = width or max(20, shutil.get_terminal_size((100, 20)).columns - 4)
    lines: list[str] = []
    for line in text.splitlines() or [text]:
        wrapped = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines[-3:]


def format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def single_line_preview(text: str, width: int | None = None) -> str:
    width = width or max(20, shutil.get_terminal_size((100, 20)).columns - 6)
    preview = " ".join(text.split())
    if len(preview) <= width:
        return preview
    return f"{preview[: max(0, width - 1)]}…"


def should_print_reasoning_summary(text: str, elapsed: float) -> bool:
    preview = " ".join(text.split())
    return bool(preview) and (
        elapsed >= _MIN_REASONING_SUMMARY_SECONDS
        or len(preview) >= _MIN_REASONING_SUMMARY_CHARS
    )


def answer_renderable(text: str) -> Table:
    layout = Table.grid(padding=(0, 1), expand=True)
    layout.add_column(width=1)
    layout.add_column(ratio=1)
    layout.add_row(Text("●", style=CLI_COLOR_ASSISTANT), Markdown(text or ""))
    return layout


class LiveMarkdownStream:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.live: Live | None = None

    def update(self, text: str) -> None:
        renderable = answer_renderable(text)
        if self.live is None:
            self.live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self.live.start(refresh=True)
            return
        self.live.update(renderable, refresh=True)

    def stop(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None


class LiveReasoningPreview:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.live: Live | None = None

    def update(self, lines: list[str]) -> None:
        text = Text("\n".join(lines), style=CLI_COLOR_THINKING)
        if self.live is None:
            self.live = Live(
                text,
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self.live.start(refresh=True)
            return
        self.live.update(text, refresh=True)

    def stop(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None


def print_startup_banner(app: Any, root: Path) -> None:
    console = Console(file=sys.stdout)
    info = app.get_model_info() if hasattr(app, "get_model_info") else {}
    model = str(info.get("model", "unknown")) if info else "unknown"
    thinking = format_thinking(info.get("thinking") if info else None)
    effort = str(info.get("reasoning_effort") or "not set") if info else "not set"
    lines = Text()
    lines.append("XCode\n", style=CLI_COLOR_TITLE)
    lines.append(f"model:    {model}\n", style=CLI_COLOR_DIM)
    lines.append(f"thinking: {thinking}\n", style=CLI_COLOR_DIM)
    lines.append(f"effort:   {effort}\n", style=CLI_COLOR_DIM)
    lines.append(f"cwd:      {root}", style=CLI_COLOR_DIM)
    console.print(
        Panel(lines, border_style=CLI_COLOR_DIM, padding=(0, 1), expand=False)
    )
    console.print(Text("Type /help for commands.", style=CLI_COLOR_DIM))


def input_prompt() -> PromptText:
    return [("class:prompt-marker", "❯ "), ("", "")]


def format_thinking(value: object) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    text = str(value).strip().lower() if value is not None else ""
    if text in {"true", "1", "yes", "on", "enabled"}:
        return "enabled"
    if text in {"false", "0", "no", "off", "disabled"}:
        return "disabled"
    return "unknown"


def create_prompt_session(
    project_root: Path | None = None,
    registry: tuple[ToolSpec, ...] = (),
    command_names: tuple[str, ...] = (),
    command_registry: dict[str, CommandEntry] | None = None,
    effort_options: Iterable[str] | Callable[[], Iterable[str]] = (),
    model_options: Iterable[str] | Callable[[], Iterable[str]] = (),
) -> PromptLike:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style
    except ImportError as exc:
        raise RuntimeError(
            "prompt_toolkit is required for REPL mode. Install it in .venv first."
        ) from exc

    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event) -> None:
        event.current_buffer.validate_and_handle()

    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    try:
        bindings.add("s-enter")(insert_newline)
    except ValueError:
        pass
    bindings.add("escape", "enter")(insert_newline)

    @bindings.add("c-c")
    def handle_ctrl_c(event) -> None:
        buf = event.current_buffer
        if buf.text:
            buf.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    completer = ReplCompleter(
        project_root or Path.cwd(),
        registry,
        command_names,
        command_registry,
        effort_options,
        model_options,
    )

    suggester = CommandArgsSuggester(completer.command_args)

    history = None
    try:
        from prompt_toolkit.history import FileHistory

        history_dir = (project_root or Path.cwd()) / ".local"
        history_dir.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_dir / "repl_history"))
    except OSError:
        pass
    style = Style.from_dict(REPL_PROMPT_STYLE)

    return PromptSessionAdapter(
        PromptSession(
            multiline=True,
            key_bindings=bindings,
            completer=completer,
            complete_while_typing=True,
            history=history,
            style=style,
            auto_suggest=suggester,
        )
    )
