"""TUI 渲染工具：Markdown→ANSI、ANSI→prompt_toolkit 片段。"""

from __future__ import annotations

import re
import shutil
from io import StringIO
from typing import cast

from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.ansi import ANSI
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def rendered_markdown_lines(text: str, width: int | None = None) -> list[str]:
    """Render markdown to plain text lines (for scroll counting)."""
    buffer = StringIO()
    Console(
        file=buffer,
        width=_markdown_width(width),
        force_terminal=False,
        color_system=None,
    ).print(Markdown(text))
    rendered = buffer.getvalue().replace("\r\n", "\n").rstrip("\n")
    return rendered.splitlines() or [""]


def markdown_ansi_lines(text: str, width: int | None = None) -> list[str]:
    """Render markdown to ANSI-colored lines for fragment rendering."""
    buffer = StringIO()
    Console(
        file=buffer,
        width=_markdown_width(width),
        force_terminal=True,
        color_system="truecolor",
    ).print(Markdown(text))
    raw = buffer.getvalue()
    lines = [line.rstrip() for line in raw.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _markdown_width(width: int | None) -> int:
    """优先使用 TUI 实际宽度，独立调用时回退到当前终端宽度。"""
    if width is not None:
        return max(1, width)
    return max(40, shutil.get_terminal_size((112, 20)).columns - 2)


def wrap_ansi_lines(lines: list[str], width: int | None) -> list[str]:
    """按显示宽度折叠 ANSI 文本，并把活动样式带到续行。"""
    if width is None or width <= 0:
        return lines
    return [wrapped for line in lines for wrapped in _wrap_ansi_line(line, width)]


def _wrap_ansi_line(line: str, width: int) -> list[str]:
    wrapped: list[str] = []
    current: list[str] = []
    active_styles: list[str] = []
    current_width = 0
    position = 0
    for match in _ANSI_ESCAPE_RE.finditer(line):
        for character in line[position : match.start()]:
            character_width = get_cwidth(character)
            if current_width and current_width + character_width > width:
                if active_styles:
                    current.append("\x1b[0m")
                wrapped.append("".join(current))
                current = list(active_styles)
                current_width = 0
            current.append(character)
            current_width += character_width
        escape = match.group(0)
        current.append(escape)
        if escape.endswith("m"):
            if escape == "\x1b[0m":
                active_styles.clear()
            else:
                active_styles.append(escape)
        position = match.end()
    for character in line[position:]:
        character_width = get_cwidth(character)
        if current_width and current_width + character_width > width:
            if active_styles:
                current.append("\x1b[0m")
            wrapped.append("".join(current))
            current = list(active_styles)
            current_width = 0
        current.append(character)
        current_width += character_width
    wrapped.append("".join(current))
    return wrapped


def render_line_fragments(line: str) -> StyleAndTextTuples:
    """Convert a single text line to styled fragments.

    If the line contains ANSI escape codes, parse them into
    prompt_toolkit style fragments.  Otherwise use simple prefix-based style.
    """
    if "\x1b[" in line:
        try:
            return cast(StyleAndTextTuples, to_formatted_text(ANSI(line)))
        except (IndexError, TypeError, ValueError):
            style = line_style(line)
            return [(style, line)]
    style = line_style(line)
    return [(style, line)]


def line_style(line: str) -> str:
    """Determine the prompt_toolkit style class for a rendered content line."""
    stripped = line.strip()
    if stripped.startswith("> /"):
        return "class:command"
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
