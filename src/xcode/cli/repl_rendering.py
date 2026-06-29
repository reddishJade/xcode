from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterable
from typing import Any, cast


from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .commands import CommandEntry, PromptLike, PromptText, ReplState
from .completion import CommandArgsSuggester, ReplCompleter
from xcode.harness.skills import ToolSpec

CITE_START = "\ue200"
CITE_SEP = "\ue202"
CITE_END = "\ue201"


_MIN_REASONING_SUMMARY_SECONDS = 0.5
_MIN_REASONING_SUMMARY_CHARS = 24

DEBUG_TOOL_RESULT_PREVIEW_LIMIT = 20_000
VERBOSE_TOOL_RESULT_PREVIEW_LIMIT = 600
NORMAL_TOOL_RESULT_PREVIEW_LIMIT = 200

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
    "scrollbar.foreground": "bg:default",
    "bottom-toolbar": "",
}


class PromptSessionAdapter:
    def __init__(self, session: PromptLike) -> None:
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


def _render_citations(text: str) -> str:
    """将 \ue200cite\ue202...\ue201 标记替换为终端可见的引用样式。"""
    pattern = (
        re.escape(CITE_START)
        + r"cite"
        + re.escape(CITE_SEP)
        + r"([^"
        + re.escape(CITE_END)
        + r"]+)"
        + re.escape(CITE_END)
    )
    return re.sub(pattern, _citation_replacement, text)


def _citation_replacement(match: re.Match[str]) -> str:
    return f"【{match.group(1)}】"


def answer_renderable(text: str) -> Table:
    rendered = _render_citations(text or "")
    layout = Table.grid(padding=(0, 1), expand=True)
    layout.add_column(width=1)
    layout.add_column(ratio=1)
    layout.add_row(Text("●", style=CLI_COLOR_ASSISTANT), Markdown(rendered))
    return layout


class LiveMarkdownStream:
    """流式渲染 agent 的 markdown 答案，支持增量更新和停止。"""

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
    """流式渲染推理过程的预览行（瞬态，完成后自动清除）。"""

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


def print_startup_banner(app: object, root: Path) -> None:
    console = Console(file=sys.stdout)
    get_model_info = getattr(app, "get_model_info", None)
    raw_info = get_model_info() if callable(get_model_info) else None
    info: dict[object, object] = raw_info if isinstance(raw_info, dict) else {}
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


def clear_terminal_display() -> None:
    """清空终端可见内容和滚动缓冲。"""
    print("\033[2J\033[3J\033[H", end="")


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


def make_bottom_toolbar(state: ReplState) -> Callable[[], str]:
    """返回 prompt_toolkit bottom_toolbar 可调用对象。"""

    def toolbar() -> str:
        parts = [f"cwd: {state.last_dir}"] if state.last_dir else []
        if state.model_name:
            parts.append(f"model: {state.model_name}")
        parts.append(f"mode: {state.mode}")
        if state.context_usage:
            parts.append(f"context: {state.context_usage}")
        if state.context_cost:
            parts.append(f"cost: {state.context_cost}")
        return "  ".join(parts)

    return toolbar


def create_prompt_session(
    project_root: Path | None = None,
    registry: tuple[ToolSpec, ...] = (),
    command_names: tuple[str, ...] = (),
    command_registry: dict[str, CommandEntry] | None = None,
    effort_options: Iterable[str] | Callable[[], Iterable[str]] = (),
    model_options: Iterable[str] | Callable[[], Iterable[str]] = (),
    skill_options: Iterable[str] | Callable[[], Iterable[str]] = (),
    state: ReplState | None = None,
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

    def handle_ctrl_c(event) -> None:
        buf = event.current_buffer
        if buf.text:
            buf.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    bindings.add("c-c")(handle_ctrl_c)

    completer = ReplCompleter(
        project_root or Path.cwd(),
        registry,
        command_names,
        command_registry,
        effort_options,
        model_options,
        skill_options,
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

    bottom_toolbar = None
    if state is not None:
        bottom_toolbar = make_bottom_toolbar(state)

    session: Any = cast(Any, PromptSession)(
        multiline=True,
        key_bindings=bindings,
        completer=completer,
        complete_while_typing=True,
        history=history,
        style=style,
        auto_suggest=suggester,
        bottom_toolbar=bottom_toolbar,
    )
    return PromptSessionAdapter(session)
