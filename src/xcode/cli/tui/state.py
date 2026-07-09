"""TUI 状态模型：_TuiState 和相关数据结构。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING

from prompt_toolkit.formatted_text import StyleAndTextTuples

from ..shared.thinking import ReasoningCore
from ..repl_tools import brief_input, final_stop_reason, tool_call_text
from .rendering import (
    markdown_ansi_lines,
    rendered_markdown_lines,
    render_line_fragments,
    visible_lines,
)

if TYPE_CHECKING:
    from xcode.agent.types import ToolInput
    from xcode.harness.agent_runtime.events import (
        CodingAgentHarnessEvent,
        FinalStructuredEvent,
    )
    from xcode.harness.observability import HITLResult


@dataclass
class _SubagentSlot:
    task: str = "waiting"
    tool: str = ""


@dataclass
class _HitlRequest:
    tool_name: str
    preview: list[str]
    event: Event
    result: HITLResult | None = None


@dataclass
class _LogEntry:
    role: str = "system"
    text: str = ""
    markdown: bool = False


@dataclass
class _TuiState:
    """TUI 核心状态。

    tool use/text_delta 事件按到达顺序直接追加到 self.log，
    和 CLI 的 Live.update() 逐事件流式输出一致。
    """

    log: list[_LogEntry] = field(default_factory=list)
    thinking_core: ReasoningCore = field(default_factory=ReasoningCore)
    subagents: dict[int, _SubagentSlot] = field(default_factory=dict)
    pending_hitl: _HitlRequest | None = None
    thinking_collapsed: bool = False
    tool_collapsed: bool = False
    running: bool = False
    mode: object = "act"

    # ── 思考快捷属性 ──

    @property
    def thinking(self) -> str:
        return self.thinking_core.text

    @thinking.setter
    def thinking(self, value: str) -> None:
        self.thinking_core.text = value

    @property
    def thinking_duration_ms(self) -> int:
        return self.thinking_core.duration_ms

    # ── 生命周期 ──

    def add_user(self, text: str) -> None:
        self.log.append(_LogEntry("you", f"> {text}"))
        self.thinking_core.reset()
        self.subagents.clear()

    def toggle_thinking(self) -> None:
        self.thinking_collapsed = not self.thinking_collapsed

    def toggle_tools(self) -> None:
        self.tool_collapsed = not self.tool_collapsed

    def handle_event(self, event: CodingAgentHarnessEvent) -> None:
        if event.type == "reasoning_delta":
            self.thinking_core.handle_delta(event.data)
        elif event.type == "text_delta":
            self._handle_text_delta(event.data)
        elif event.type == "tool_use":
            self._record_tool_use(event.data.id, event.data.name, event.data.input)
        elif event.type == "tool_update":
            self._handle_tool_update(event.data.tool_name, event.data.partial_result)
        elif event.type == "tool_result":
            self._record_tool_result(
                event.data.tool_use_id,
                event.data.status,
                event.data.content,
            )
        elif event.type == "final":
            self._finish_answer(event)

    def render(self) -> str:
        return "\n".join(self.lines()).rstrip() + "\n"

    # ── 片段生成 ──

    def fragments(
        self, limit: int | None = None, scrollback: int = 0
    ) -> StyleAndTextTuples:
        all_ansi = self.ansi_lines()
        visible = visible_lines(all_ansi, limit, scrollback)
        result: StyleAndTextTuples = []
        for line in visible:
            result.extend(render_line_fragments(line))
            result.append(("", "\n"))
        return result

    def top_bar(self, project_name: str) -> str:
        state = "busy" if self.running else "idle"
        return f" Xcode  ·  {state}  ·  mode {self.mode}  ·  cwd {project_name} "

    def status(self, scrollback: int = 0) -> str:
        state = "busy" if self.running else "idle"
        thinking = "thinking:collapsed" if self.thinking_collapsed else "thinking:open"
        tools = "tools:collapsed" if self.tool_collapsed else "tools:open"
        scroll = f" · scroll {scrollback}" if scrollback else ""
        return f" {state} · Ctrl-T {thinking} · Ctrl-O {tools}{scroll} "

    # ── 文本渲染（纯文本：用于滚动高度计算） ──

    def lines(self) -> list[str]:
        lines: list[str] = []
        for entry in self.log:
            self._append_entry_lines(entry, lines, rendered_markdown_lines)
        if self.thinking.strip():
            self._thinking_lines(lines)
        self._append_subagent_lines(lines)
        self._append_hitl_lines(lines)
        return lines

    # ── ANSI 渲染（彩色：用于片段生成） ──

    def ansi_lines(self) -> list[str]:
        lines: list[str] = []
        for entry in self.log:
            self._append_entry_lines(entry, lines, markdown_ansi_lines)
        if self.thinking.strip():
            self._thinking_lines(lines)
        elif self.running:
            if lines:
                lines.append("")
            lines.append("│ thinking")
        self._append_subagent_lines(lines)
        self._append_hitl_lines(lines)
        return lines

    # ── 内部渲染方法 ──

    def _append_entry_lines(
        self,
        entry: _LogEntry,
        lines: list[str],
        md_fn: Callable[[str], list[str]],
    ) -> None:
        if lines:
            lines.append("")
        if not entry.role:
            lines.extend(entry.text.splitlines())
        elif entry.role == "thinking":
            self._append_thinking_entry(entry, lines, plain_text=(md_fn is rendered_markdown_lines))
        elif entry.markdown:
            lines.append(f"│ {entry.role}")
            for line in md_fn(entry.text):
                lines.append(f"│   {line}")
        else:
            lines.append(f"│ {entry.role}")
            for line in entry.text.splitlines() or [""]:
                lines.append(f"│   {line}")

    def _thinking_lines(self, lines: list[str]) -> None:
        if lines:
            lines.append("")
        dur = self.thinking_core.duration_ms
        if dur:
            lines.append(f"│ thought for {dur}ms")
        else:
            lines.append("│ thinking")
        if not self.thinking_collapsed:
            for tl in self.thinking.splitlines():
                lines.append(f"│   {tl.lstrip()}")

    def _append_thinking_entry(
        self, entry: _LogEntry, lines: list[str], plain_text: bool
    ) -> None:
        entry_lines = entry.text.splitlines() or [""]
        has_timing = any("Thought for" in ln for ln in entry_lines)
        dur_text = entry_lines[-1].strip() if has_timing else ""
        if has_timing:
            entry_lines = entry_lines[:-1]
        if dur_text:
            dur_ms = dur_text.split()[-1].rstrip("ms")
            lines.append(f"│ thought for {dur_ms}ms")
        else:
            lines.append("│ thinking")
        if not self.thinking_collapsed:
            for tl in entry_lines:
                if plain_text:
                    lines.append(f"│   {tl}")
                else:
                    lines.append(f"│   {tl.lstrip()}")

    def _append_subagent_lines(self, lines: list[str]) -> None:
        if not self.subagents:
            return
        if lines:
            lines.append("")
        lines.append("│ subagents")
        for index in sorted(self.subagents):
            slot = self.subagents[index]
            lines.append(f"│   [{index}] {slot.task}")
            if slot.tool:
                lines.append(f"│       {slot.tool}")

    def _append_hitl_lines(self, lines: list[str]) -> None:
        if self.pending_hitl is None:
            return
        if lines:
            lines.append("")
        lines.append("│ authorization request")
        for line in self.pending_hitl.preview:
            lines.append(f"│   {line}")

    # ── 工具事件记录（直接追加到 log） ──

    def _handle_text_delta(self, delta: str) -> None:
        """追加 text_delta 到 log 的最后一个 xcode 条目（或在流式期间追加新行）。"""
        # 追加增量文本到当前答句末尾
        self._append_or_update_answer(delta)

    def _append_or_update_answer(self, delta: str) -> None:
        if self.log and self.log[-1].role == "xcode" and self.log[-1].markdown:
            self.log[-1] = _LogEntry("xcode", self.log[-1].text + delta, markdown=True)
        else:
            self.log.append(_LogEntry("xcode", delta, markdown=True))

    def _record_tool_use(self, tool_id: str, name: str, raw_input: ToolInput) -> None:
        label = brief_input(name, raw_input)
        self.log.append(_LogEntry("", f"│ tool {label}"))
        if name in {"todowrite", "subagent"}:
            text = tool_call_text(name, label, raw_input).plain
            self.log.append(_LogEntry("", f"│   {text.strip()}"))

    def _record_tool_result(
        self, tool_id: str, status: str, content: str
    ) -> None:
        from .rendering import tail_line

        if status == "ok":
            summary = tail_line(content)
            detail = f"✓ {summary}" if summary else "✓"
            self.log.append(_LogEntry("", f"│   {detail}"))
            return
        self.log.append(_LogEntry("", f"│ ✗ {content}"))

    def _handle_tool_update(self, tool_name: str, partial: str) -> None:
        from .rendering import tail_line

        if tool_name == "subagent":
            for line in partial.splitlines():
                self._record_subagent_update(line.strip())
            return
        clean = tail_line(partial)
        if clean:
            self.log.append(_LogEntry("", f"│   {clean}"))

    def record_subagent_update(self, clean: str) -> bool:
        """公开的 subagent 更新入口，测试和 _handle_tool_update 共用。"""
        return self._record_subagent_update(clean)

    def _record_subagent_update(self, clean: str) -> bool:
        if not clean:
            return False
        match = re.match(r"\[(\d+)]( +)(.*)", clean)
        if match is None:
            return False
        index = int(match.group(1))
        gap = match.group(2)
        body = match.group(3)
        slot = self.subagents.setdefault(index, _SubagentSlot())
        if len(gap) > 1:
            slot.tool = body.strip()
        else:
            slot.task = body
            if body.startswith(("✓", "✗")):
                slot.tool = ""
        return True

    # ── 回合结束 ──

    def _finish_answer(self, event: FinalStructuredEvent) -> None:
        # 持久化思考（自动包含耗时信息）
        if self.thinking.strip():
            self.thinking_core.finish()
            text = self.thinking.strip()
            if self.thinking_duration_ms:
                text += f"\nThought for {self.thinking_duration_ms}ms"
            self.log.append(_LogEntry("thinking", text))

        # 追加 final 事件的答案（如果 log 中最后的 xcode 条目不完整）
        final_answer = event.data.answer.strip()
        if final_answer and (
            not self.log
            or self.log[-1].role != "xcode"
            or not self.log[-1].text.strip().endswith(final_answer)
        ):
            self.log.append(_LogEntry("xcode", final_answer, markdown=True))

        reason = final_stop_reason(event.data)
        if reason:
            self.log.append(_LogEntry("stop", reason))
        self.thinking_core.reset()
        self.running = False
