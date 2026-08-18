"""CLI 与 TUI 共享的类型化工具结果呈现。"""

from __future__ import annotations

from xcode.agent.types import (
    DiffRenderIntent,
    TerminalRenderIntent,
    ToolRenderIntent,
)

from .shared.thinking import single_line_preview


def render_intent_summary(
    intent: ToolRenderIntent,
    content: str = "",
) -> str:
    """将呈现意图投影为紧凑、宿主无关的单行摘要。"""
    if isinstance(intent, TerminalRenderIntent):
        output = single_line_preview(content.rstrip()) if content.strip() else "done"
        return f"Terminal {intent.cwd}: {output}"
    if isinstance(intent, DiffRenderIntent):
        targets = ", ".join(intent.files) or "workspace"
        line = (
            f":{intent.first_changed_line}"
            if intent.first_changed_line is not None
            else ""
        )
        return f"Changed {targets}{line}"
    if intent.line_start is None:
        return intent.path
    if intent.line_end is None or intent.line_end == intent.line_start:
        return f"{intent.path}:{intent.line_start}"
    return f"{intent.path}:{intent.line_start}-{intent.line_end}"
