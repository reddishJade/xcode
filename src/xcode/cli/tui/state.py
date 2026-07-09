"""TUI 状态模型：_TuiState 和相关数据结构。"""

from __future__ import annotations

import re
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
    tool_block,
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
class _ToolSlot:
    name: str
    label: str
    text: str


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
    """TUI 核心状态，包含完整的对话渲染逻辑。

    不绑定 prompt_toolkit 或 Rich，输出由上层 app.py 分派到 Label/Window。
    """

    log: list[_LogEntry] = field(default_factory=list)
    current_answer: str = ""
    thinking_core: ReasoningCore = field(default_factory=ReasoningCore)
    tool_events: list[str] = field(default_factory=list)
    tool_labels: dict[str, _ToolSlot] = field(default_factory=dict)
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
        self.current_answer = ""
        self.thinking_core.reset()
        self._clear_activity()

    def toggle_thinking(self) -> None:
        self.thinking_collapsed = not self.thinking_collapsed

    def toggle_tools(self) -> None:
        self.tool_collapsed = not self.tool_collapsed

    def handle_event(self, event: CodingAgentHarnessEvent) -> None:
        if event.type == "reasoning_delta":
            self.thinking_core.handle_delta(event.data)
        elif event.type == "text_delta":
            self.current_answer += event.data
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
            if lines:
                lines.append("")
            if not entry.role:
                lines.extend(entry.text.splitlines())
            elif entry.role == "thinking":
                self._append_thinking_entry(entry, lines, plain_text=True)
            elif entry.markdown:
                lines.append(f"│ {entry.role}")
                for line in rendered_markdown_lines(entry.text):
                    lines.append(f"│   {line}")
            else:
                lines.append(f"│ {entry.role}")
                for line in entry.text.splitlines() or [""]:
                    lines.append(f"│   {line}")
        if self.thinking.strip():
            self._thinking_lines(lines)
        self._append_activity_lines(lines)
        self._append_hitl_lines(lines)
        if self.current_answer.strip():
            if lines:
                lines.append("")
            lines.append("│ xcode")
            lines.extend(rendered_markdown_lines(self.current_answer.strip()))
        return lines

    # ── ANSI 渲染（彩色：用于片段生成） ──

    def ansi_lines(self) -> list[str]:
        lines: list[str] = []
        for entry in self.log:
            if lines:
                lines.append("")
            if not entry.role:
                lines.extend(entry.text.splitlines())
            elif entry.role == "thinking":
                self._append_thinking_entry(entry, lines, plain_text=False)
            elif entry.markdown:
                lines.append(f"│ {entry.role}")
                for line in markdown_ansi_lines(entry.text):
                    lines.append(f"│   {line}")
            else:
                lines.append(f"│ {entry.role}")
                for line in entry.text.splitlines() or [""]:
                    lines.append(f"│   {line}")
        if self.thinking.strip():
            self._thinking_lines(lines)
        elif self.running and not self.current_answer.strip():
            if lines:
                lines.append("")
            lines.append("│ thinking")
        self._append_activity_lines(lines)
        self._append_hitl_lines(lines)
        if self.current_answer.strip():
            if lines:
                lines.append("")
            lines.append("│ xcode")
            for line in markdown_ansi_lines(self.current_answer.strip()):
                lines.append(f"│   {line}")
        return lines

    # ── 内部渲染方法 ──

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

    def _append_activity_lines(self, lines: list[str]) -> None:
        if self.tool_collapsed and (self.tool_events or self.subagents):
            if lines:
                lines.append("")
            lines.append("│ tools collapsed")
            return
        for event in self.tool_events[-40:]:
            if lines:
                lines.append("")
            lines.extend(event.splitlines())
        if self.subagents:
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
        lines.append("│   Deny | Allow (once) | Allow this session | Always allow")

    # ── 工具事件记录 ──

    def _record_tool_use(self, tool_id: str, name: str, raw_input: ToolInput) -> None:
        label = brief_input(name, raw_input)
        text = tool_call_text(name, label, raw_input).plain
        self.tool_labels[tool_id] = _ToolSlot(name=name, label=label, text=text)
        self.tool_events.append(tool_block(name, "running", f"正在调用工具: {label}"))
        if name in {"todowrite", "subagent"}:
            self.tool_events.append(tool_block(name, "input", text.strip()))

    def _record_tool_result(
        self, tool_id: str, status: str, content: str
    ) -> None:
        slot = self.tool_labels.get(tool_id)
        name = slot.name if slot else tool_id
        label = slot.label if slot else tool_id
        if status == "ok":
            from .rendering import tail_line

            summary = tail_line(content)
            detail = f"✓ {label}" + (f" -> {summary}" if summary else "")
            self.tool_events.append(tool_block(name, "success", detail))
            return
        self.tool_events.append(tool_block(name, "error", f"✗ {label}: {content}"))

    def _handle_tool_update(self, tool_name: str, partial: str) -> None:
        from .rendering import tail_line

        if tool_name != "subagent":
            clean = tail_line(partial)
            if clean:
                self.tool_events.append(tool_block(tool_name, "update", clean))
            return
        for line in partial.splitlines():
            self.record_subagent_update(line.strip())

    def record_subagent_update(self, clean: str) -> bool:
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
        answer = self.current_answer.strip() or event.data.answer.strip()

        # Persist thinking before clearing
        if self.thinking.strip():
            self.thinking_core.finish()
            text = self.thinking.strip()
            if self.thinking_duration_ms:
                text += f"\nThought for {self.thinking_duration_ms}ms"
            self.log.append(_LogEntry("thinking", text))

        # Persist tool activity before clearing
        activity_lines: list[str] = []
        for ev in self.tool_events[-40:]:
            if activity_lines:
                activity_lines.append("")
            activity_lines.extend(ev.splitlines())
        activity = "\n".join(activity_lines).strip()
        self._clear_activity()
        if activity:
            self.log.append(_LogEntry(role="", text=activity))
        if answer:
            self.log.append(_LogEntry("xcode", answer, markdown=True))
        reason = final_stop_reason(event.data)
        if reason:
            self.log.append(_LogEntry("stop", reason))
        self.current_answer = ""
        self.thinking_core.reset()
        self.running = False

    def _clear_activity(self) -> None:
        self.tool_events.clear()
        self.tool_labels.clear()
        self.subagents.clear()
