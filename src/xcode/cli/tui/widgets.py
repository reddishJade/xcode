"""TUI 自定义 prompt_toolkit 组件：输入栏高亮器、伪 PromptSession。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer

from ..commands import PromptText

_file_ref_pattern = re.compile(r"(?<!\S)@([^\s]+)")


class TuiInputLexer(Lexer):
    """高亮 TUI 输入栏中的 ! 前缀和 @file 引用。"""

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
            return self._highlight(lines[line_number], line_number == 0)

        return get_line

    @staticmethod
    def _highlight(line: str, first_line: bool) -> StyleAndTextTuples:
        frags: list[tuple[str, str]] = []
        cursor = 0
        if first_line and line.startswith("!"):
            frags.append(("fg:ansiyellow bold", "!"))
            cursor = 1
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


class TuiPromptSession:
    """TUI 占位 PromptSession——CLI 的 PromptSession 适配接口。

    实际 TUI 使用 TextArea 输入，不需要 CLI 的 PromptSession。
    """

    def prompt(self, prompt_text: PromptText) -> str:
        _ = prompt_text
        return ""
