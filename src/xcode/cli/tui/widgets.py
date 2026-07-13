"""TUI 自定义 prompt_toolkit 组件：输入栏高亮器、伪 PromptSession。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from ..commands import PromptText

_file_ref_pattern = re.compile(r"(?<!\S)@([^\s]+)")


class TuiOutputControl(FormattedTextControl):
    """处理 inline TUI 输出区域的鼠标滚轮。"""

    def __init__(self, on_scroll: Callable[[int], None]) -> None:
        super().__init__(text="", focusable=False)
        self._on_scroll = on_scroll

    def mouse_handler(self, mouse_event: MouseEvent) -> object:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._on_scroll(3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._on_scroll(-3)
            return None
        return super().mouse_handler(mouse_event)


class TuiInputLexer(Lexer):
    """高亮 TUI 输入栏中的 @file 引用。"""

    def lex_document(self, document: object) -> Callable[[int], StyleAndTextTuples]:
        lines: list[str] = (
            getattr(document, "lines", None)
            or str(getattr(document, "text", "")).splitlines()
        )
        if not lines:
            lines = [""]

        def get_line(line_number: int) -> StyleAndTextTuples:
            if line_number < 0 or line_number >= len(lines):
                return []
            return self._highlight(lines[line_number])

        return get_line

    @staticmethod
    def _highlight(line: str) -> StyleAndTextTuples:
        frags: list[tuple[str, str]] = []
        cursor = 0
        for m in _file_ref_pattern.finditer(line, cursor):
            if m.start() > cursor:
                frags.append(("", line[cursor : m.start()]))
            frags.append(("fg:ansicyan bold", m.group(0)))
            cursor = m.end()
        if cursor < len(line):
            frags.append(("", line[cursor:]))
        if not frags:
            frags.append(("", line))
        return cast(StyleAndTextTuples, frags)


def tui_input_prompt(
    awaiting_denial_suggestion: bool, is_shell_command: bool
) -> StyleAndTextTuples:
    """返回输入提示符；shell 命令模式时强调输入标记。"""
    marker_style = "class:prompt-marker" if is_shell_command else ""
    if awaiting_denial_suggestion:
        return [("", "Tell model what to do "), (marker_style, "> ")]
    return [(marker_style, "> ")]


class TuiPromptSession:
    """TUI 占位 PromptSession——CLI 的 PromptSession 适配接口。

    实际 TUI 使用 TextArea 输入，不需要 CLI 的 PromptSession。
    """

    def prompt(self, prompt_text: PromptText) -> str:
        _ = prompt_text
        return ""
