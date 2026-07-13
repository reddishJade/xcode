"""TUI 渲染工具：Markdown→ANSI、ANSI→prompt_toolkit 片段。"""

from __future__ import annotations

from io import StringIO
import shutil
from typing import cast

from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.ansi import ANSI
from rich.console import Console
from rich.markdown import Markdown


def rendered_markdown_lines(text: str) -> list[str]:
    """Render markdown to plain text lines (for scroll counting)."""
    buffer = StringIO()
    Console(
        file=buffer,
        width=max(40, shutil.get_terminal_size((112, 20)).columns - 2),
        force_terminal=False,
        color_system=None,
    ).print(Markdown(text))
    rendered = buffer.getvalue().replace("\r\n", "\n").rstrip("\n")
    return rendered.splitlines() or [""]


def markdown_ansi_lines(text: str) -> list[str]:
    """Render markdown to ANSI-colored lines for fragment rendering."""
    buffer = StringIO()
    Console(
        file=buffer,
        width=max(40, shutil.get_terminal_size((112, 20)).columns - 2),
        force_terminal=True,
        color_system="truecolor",
    ).print(Markdown(text))
    raw = buffer.getvalue()
    lines = [line.rstrip() for line in raw.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def render_line_fragments(line: str) -> StyleAndTextTuples:
    """Convert a single text line to styled fragments.

    If the line contains ANSI escape codes, parse them into
    prompt_toolkit style fragments.  Otherwise use simple prefix-based style.
    """
    if "\x1b[" in line:
        try:
            return cast(StyleAndTextTuples, to_formatted_text(ANSI(line)))
        except Exception:
            pass
    style = line_style(line)
    return [(style, line)]


def line_style(line: str) -> str:
    """Determine the prompt_toolkit style class for a rendered content line."""
    stripped = line.strip()
    if stripped.startswith("> "):
        return "class:user"
    if stripped.startswith(("Thinking", "Thought for")):
        return "class:thinking"
    if stripped.startswith(("● ", "• Exploring", "• Explored")):
        return "class:tool-title"
    if stripped.startswith("?"):
        return "class:error"
    if stripped.startswith(("└ ", "⎿")):
        return "class:tool"
    if stripped.startswith("─"):
        return "class:border"
    if "✗" in stripped:
        return "class:error"
    return ""


def visible_lines(lines: list[str], limit: int | None, scrollback: int) -> list[str]:
    if limit is None or len(lines) <= limit:
        return lines
    end = max(limit, len(lines) - scrollback)
    start = max(0, end - limit)
    return lines[start:end]
