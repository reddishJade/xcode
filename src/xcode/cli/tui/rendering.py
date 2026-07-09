"""TUI 渲染工具：Markdown→ANSI、ANSI→prompt_toolkit 片段。"""

from __future__ import annotations

from io import StringIO
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
        width=112,
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
        width=112,
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
    if stripped in {"│ you", "you"} or stripped.startswith("│ you"):
        return "class:user"
    if stripped in {"│ xcode", "xcode"} or stripped.startswith("│ xcode"):
        return "class:assistant"
    if stripped.startswith("│ thinking"):
        return "class:thinking"
    if stripped.startswith((
        "│ tool", "│ subagents", "│ tools collapsed", "│ authorization"
    )):
        return "class:tool-title"
    if stripped.startswith("│ permission requested"):
        return "class:error"
    if stripped.startswith(("│   [", "│       ", "│   ✓", "│   <-")):
        return "class:tool"
    if stripped.startswith("│ error") or "✗" in stripped:
        return "class:error"
    if stripped.startswith("│"):
        return ""
    return ""


def tail_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def visible_lines(
    lines: list[str], limit: int | None, scrollback: int
) -> list[str]:
    if limit is None or len(lines) <= limit:
        return lines
    end = max(limit, len(lines) - scrollback)
    start = max(0, end - limit)
    return lines[start:end]


def hitl_preview_lines(tool_name: str, tool_input: dict) -> list[str]:
    """为 HITL 弹窗生成工具调用预览行。

    CLI 同等等效：repl_hitl._print_tool_preview() 但输出为文本行而非 Rich Panel。
    """
    from ..repl_tools import brief_input

    lines = [f"Tool: {tool_name}"]
    if tool_name == "edit_file":
        path = tool_input.get("path", "")
        lines.append(f"File: {path}")
        if tool_input.get("replace_all", False):
            lines.append("Replace all occurrences")
        old_text = str(tool_input.get("old_text", ""))
        new_text = str(tool_input.get("new_text", ""))
        if old_text:
            lines.append(f"- {old_text[:280].replace(chr(10), '¶ ')}")
        if new_text:
            lines.append(f"+ {new_text[:280].replace(chr(10), '¶ ')}")
    elif tool_name == "bash":
        command = str(tool_input.get("command") or tool_input.get("input", ""))
        lines.append(f"Command: {command[:500] or '(empty)'}")
        parts = command.strip().split()
        if parts:
            lines.append(f"Command type: {parts[0].lower()}")
    elif tool_name == "write_file":
        path = tool_input.get("path", "")
        content = str(tool_input.get("content", ""))
        lines.append(f"File: {path}")
        if content:
            lines.append(f"Content: {len(content.splitlines())} lines")
    elif tool_name == "read_file":
        lines.append(f"File: {tool_input.get('path', '')}")
    elif tool_name in {"grep_search", "glob_files", "find_files"}:
        pattern = (
            tool_input.get("pattern")
            or tool_input.get("query")
            or tool_input.get("path", "")
        )
        lines.append(f"Pattern: {str(pattern)[:200]}")
        if tool_input.get("path") or tool_input.get("include"):
            lines.append(
                f"Search in: {tool_input.get('path') or tool_input.get('include')}"
            )
    else:
        lines.append(f"Input: {brief_input(tool_name, tool_input)}")
    return lines
