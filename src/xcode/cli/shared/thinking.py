"""输出无关的推理过程核心逻辑——CLI 和 TUI 共享。"""

from __future__ import annotations

import shutil
import textwrap
import time


_MIN_REASONING_SUMMARY_SECONDS = 0.5
_MIN_REASONING_SUMMARY_CHARS = 24


def format_elapsed(seconds: float) -> str:
    """推理耗时格式化。"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def should_print_reasoning_summary(text: str, elapsed: float) -> bool:
    """判断是否有足够的推理内容值得显示摘要。"""
    preview = " ".join(text.split())
    return bool(preview) and (
        elapsed >= _MIN_REASONING_SUMMARY_SECONDS
        or len(preview) >= _MIN_REASONING_SUMMARY_CHARS
    )


def reasoning_preview_lines(text: str, width: int | None = None) -> list[str]:
    """从完整推理文本中截取最后几行作为实时预览。"""
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


def single_line_preview(text: str, width: int | None = None) -> str:
    """将任意文本压缩到单行预览。"""
    width = width or max(20, shutil.get_terminal_size((100, 20)).columns - 6)
    preview = " ".join(text.split())
    if len(preview) <= width:
        return preview
    return f"{preview[: max(0, width - 1)]}…"


class ReasoningCore:
    """输出无关的推理过程追踪。

    累积推理增量文本、计时、生成摘要。
    不绑定 Rich/Live，CLI 和 TUI 都可复用。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.started_at: float | None = None
        self.duration_ms: int = 0

    def handle_delta(self, data: str) -> None:
        if self.started_at is None:
            self.started_at = time.perf_counter()
        self.text += data

    def finish(self) -> None:
        if self.started_at is None:
            return
        elapsed = time.perf_counter() - self.started_at
        self.duration_ms = int(elapsed * 1000)

    @property
    def preview_lines(self) -> list[str]:
        return reasoning_preview_lines(self.text)
