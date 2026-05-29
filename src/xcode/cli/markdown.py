from __future__ import annotations

import sys
from typing import Protocol


class MarkdownRenderer(Protocol):
    def render(self, text: str) -> None: ...


class TerminalMarkdownRenderer:
    """在终端中渲染 Markdown，缺少 rich 时回退为原始文本。"""

    def render(self, text: str) -> None:
        if not text:
            return
        try:
            from rich.console import Console
            from rich.markdown import Markdown
        except ImportError:
            print(text)
            return

        console = Console(file=sys.stdout)
        console.print(Markdown(text))
