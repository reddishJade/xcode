"""TUI 状态模型：_TuiState 和相关数据结构。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, cast

from prompt_toolkit.formatted_text import StyleAndTextTuples

from ..shared.thinking import ReasoningCore, format_elapsed, single_line_preview
from ..repl_tools import brief_input, final_stop_reason, tool_call_text
from ..repl_rendering import _render_citations
from xcode.agent.types import ToolInput
from .rendering import (
    markdown_ansi_lines,
    rendered_markdown_lines,
    render_line_fragments,
    visible_lines,
)
from xcode.harness.observability.shell_analyzer import analyze_shell_command


_THINKING_ANSI = "\x1b[38;2;128;128;128m"
_ANSI_RESET = "\x1b[0m"

if TYPE_CHECKING:
    from xcode.harness.agent_runtime.events import (
        CodingAgentHarnessEvent,
        FinalStructuredEvent,
    )
    from xcode.harness.observability import HITLResult
    from xcode.harness.session import SessionEntry


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
class _ExplorationCall:
    tool_id: str
    label: str
    complete: bool = False
    failed: bool = False


@dataclass
class _LogEntry:
    role: str = "system"
    text: str = ""
    markdown: bool = False
    exploration_calls: list[_ExplorationCall] = field(default_factory=list)


@dataclass
class _TuiState:
    """TUI 核心状态。

    tool use/text_delta 事件按到达顺序直接追加到 self.log，
    和 CLI 的 Live.update() 逐事件流式输出一致。
    """

    log: list[_LogEntry] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)
    project_root: Path | None = None
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
        self.log.append(_LogEntry("you", text))
        self.thinking_core.reset()
        self.subagents.clear()

    def toggle_thinking(self) -> None:
        self.thinking_collapsed = not self.thinking_collapsed

    def toggle_tools(self) -> None:
        self.tool_collapsed = not self.tool_collapsed

    def restore_history(self, records: list[SessionEntry]) -> None:
        """将会话分支重建为与实时输出一致的 TUI 日志。"""
        self.log.clear()
        self.tool_names.clear()
        self.thinking_core.reset()
        self.subagents.clear()
        self.pending_hitl = None
        self.running = False
        for record in records:
            if record.type == "user":
                self.add_user(str(record.content))
            elif record.type == "assistant":
                text = str(record.content).strip()
                if text:
                    self.log.append(_LogEntry("xcode", text, markdown=True))
            elif record.type == "event" and isinstance(record.content, Mapping):
                self._restore_event(record.content)

    def _restore_event(self, content: Mapping[str, object]) -> None:
        event_type = content.get("type")
        data = content.get("data")
        if event_type == "thinking":
            self._restore_thinking(data)
        elif event_type == "tool_use" and isinstance(data, dict):
            self._record_tool_use(
                str(data.get("id", "")),
                str(data.get("name", "tool")),
                cast(
                    ToolInput,
                    data.get("input") if isinstance(data.get("input"), dict) else {},
                ),
            )
        elif event_type == "tool_result" and isinstance(data, dict):
            self._record_tool_result(
                str(data.get("tool_use_id", "")),
                str(data.get("status", "ok")),
                str(data.get("content", "")),
            )

    def _restore_thinking(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        text = str(data.get("content", "")).strip()
        if not text:
            return
        duration = data.get("duration_ms")
        suffix = (
            f"\nThought for {format_elapsed(duration / 1000)}"
            if isinstance(duration, int) and duration > 0
            else ""
        )
        self.log.append(_LogEntry("thinking", text + suffix))

    def handle_event(self, event: CodingAgentHarnessEvent) -> None:
        if event.type == "reasoning_delta":
            self.thinking_core.handle_delta(event.data)
        elif event.type == "text_delta":
            self._handle_text_delta(event.data)
        elif event.type == "tool_use":
            self._finish_thinking()
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

    # ── 文本渲染（纯文本：用于滚动高度计算） ──

    def lines(self) -> list[str]:
        lines: list[str] = []
        self._append_log_entries(lines, rendered_markdown_lines, color_thinking=False)
        if self.thinking.strip():
            self._thinking_lines(lines, color=False)
        self._append_subagent_lines(lines)
        self._append_hitl_lines(lines)
        return lines

    # ── ANSI 渲染（彩色：用于片段生成） ──

    def ansi_lines(self) -> list[str]:
        lines: list[str] = []
        self._append_log_entries(lines, markdown_ansi_lines, color_thinking=True)
        if self.thinking.strip():
            self._thinking_lines(lines, color=True)
        self._append_subagent_lines(lines)
        self._append_hitl_lines(lines)
        return lines

    # ── 内部渲染方法 ──

    def _append_log_entries(
        self,
        lines: list[str],
        md_fn: Callable[[str], list[str]],
        color_thinking: bool,
    ) -> None:
        """按工具组渲染日志，并只在当前工具组显示折叠提示。"""
        latest_tool_index = max(
            (
                index
                for index, entry in enumerate(self.log)
                if entry.role in {"tool", "exploration"}
            ),
            default=-1,
        )
        latest_detail_index = max(
            (
                index
                for index, entry in enumerate(self.log)
                if index > latest_tool_index
                and entry.role in {"tool-detail", "exploration"}
            ),
            default=-1,
        )
        for index, entry in enumerate(self.log):
            self._append_entry_lines(
                entry,
                lines,
                md_fn,
                color_thinking=color_thinking,
                show_tool_expand=(self.tool_collapsed and index == latest_tool_index),
                show_tool_collapse=(
                    not self.tool_collapsed
                    and (
                        index == latest_detail_index
                        or (entry.role == "exploration" and index == latest_tool_index)
                    )
                ),
            )

    def _append_entry_lines(
        self,
        entry: _LogEntry,
        lines: list[str],
        md_fn: Callable[[str], list[str]],
        color_thinking: bool,
        show_tool_expand: bool,
        show_tool_collapse: bool,
    ) -> None:
        if entry.role == "tool-detail" and self.tool_collapsed:
            return
        if lines and entry.role not in {"tool", "tool-detail", "exploration"}:
            lines.append("")
        if entry.role == "you":
            user_lines = entry.text.splitlines() or [""]
            lines.append(f"> {user_lines[0]}")
            lines.extend(f"  {line}" for line in user_lines[1:])
        elif entry.role == "tool":
            suffix = " (ctrl+o to expand)" if show_tool_expand else ""
            lines.append(f"{entry.text}{suffix}")
        elif entry.role == "exploration":
            active = any(not call.complete for call in entry.exploration_calls)
            title = "• Exploring" if active else "• Explored"
            if self.tool_collapsed:
                suffix = " (ctrl+o to expand)" if show_tool_expand else ""
                lines.append(f"{title}{suffix}")
                return
            lines.append(title)
            for call in entry.exploration_calls:
                suffix = " — failed" if call.failed else ""
                lines.append(f"  └ {call.label}{suffix}")
            if show_tool_collapse and entry.exploration_calls:
                lines[-1] += " (ctrl+o to collapse)"
        elif entry.role in {"", "tool-detail"}:
            detail_lines = entry.text.splitlines() or [""]
            if show_tool_collapse:
                detail_lines[-1] += " (ctrl+o to collapse)"
            lines.extend(detail_lines)
        elif entry.role == "thinking":
            self._append_thinking_entry(
                entry,
                lines,
                plain_text=(md_fn is rendered_markdown_lines),
                color=color_thinking,
            )
        elif entry.markdown:
            lines.extend(md_fn(_render_citations(entry.text)))
        else:
            lines.extend(entry.text.splitlines() or [""])

    def _thinking_lines(self, lines: list[str], color: bool) -> None:
        if lines:
            lines.append("")
        dur = self.thinking_core.duration_ms
        if dur:
            self._append_thinking_line(
                lines, f"Thought for {format_elapsed(dur / 1000)}", color
            )
        else:
            self._append_thinking_line(lines, "Thinking", color)
        if not self.thinking_collapsed:
            for tl in self.thinking.splitlines():
                self._append_thinking_line(lines, f"  {tl.lstrip()}", color)

    def _append_thinking_entry(
        self, entry: _LogEntry, lines: list[str], plain_text: bool, color: bool
    ) -> None:
        entry_lines = entry.text.splitlines() or [""]
        has_timing = entry_lines[-1].startswith("Thought for ")
        dur_text = entry_lines[-1].removeprefix("Thought for ") if has_timing else ""
        if has_timing:
            entry_lines = entry_lines[:-1]
        if dur_text:
            self._append_thinking_line(lines, f"Thought for {dur_text}", color)
        else:
            self._append_thinking_line(lines, "Thinking", color)
        if not self.thinking_collapsed:
            for tl in entry_lines:
                self._append_thinking_line(
                    lines, f"  {tl if plain_text else tl.lstrip()}", color
                )

    @staticmethod
    def _append_thinking_line(lines: list[str], text: str, color: bool) -> None:
        if color:
            lines.append(f"{_THINKING_ANSI}{text}{_ANSI_RESET}")
        else:
            lines.append(text)

    def _append_subagent_lines(self, lines: list[str]) -> None:
        if not self.subagents or self.tool_collapsed:
            return
        if lines:
            lines.append("")
        lines.append("● Subagents")
        for index in sorted(self.subagents):
            slot = self.subagents[index]
            lines.append(f"  └ [{index}] {slot.task}")
            if slot.tool:
                lines.append(f"      {slot.tool}")

    def _append_hitl_lines(self, lines: list[str]) -> None:
        if self.pending_hitl is None:
            return
        if lines:
            lines.append("")
        lines.append("? Authorization request")
        for line in self.pending_hitl.preview:
            lines.append(f"  {line}")

    # ── 工具事件记录（直接追加到 log） ──

    def _handle_text_delta(self, delta: str) -> None:
        """追加 text_delta 到 log 的最后一个 xcode 条目（或在流式期间追加新行）。"""
        self._finish_thinking()
        self._append_or_update_answer(delta)

    def _append_or_update_answer(self, delta: str) -> None:
        if self.log and self.log[-1].role == "xcode" and self.log[-1].markdown:
            self.log[-1] = _LogEntry("xcode", self.log[-1].text + delta, markdown=True)
        else:
            self.log.append(_LogEntry("xcode", delta, markdown=True))

    def _record_tool_use(self, tool_id: str, name: str, raw_input: ToolInput) -> None:
        if _is_exploration_call(name, raw_input):
            self._record_exploration_call(
                _ExplorationCall(tool_id, _exploration_label(name, raw_input))
            )
            return
        self.tool_names[tool_id] = name
        label = brief_input(name, raw_input)
        if name == "list_dir":
            path = Path(str(raw_input.get("path", ".")))
            if self.project_root is not None and not path.is_absolute():
                path = self.project_root / path
            label = f"ListDir({path.as_posix()})"
        elif name == "read_file":
            path = Path(str(raw_input.get("path", "")))
            if self.project_root is not None and not path.is_absolute():
                path = self.project_root / path
            limit = raw_input.get("limit")
            label = f"Read({path.as_posix()})" + (f" ({limit} lines)" if limit else "")
        else:
            label = label[:1].upper() + label[1:]
        self.log.append(_LogEntry("tool", f"● {label}"))
        if name in {"todowrite", "subagent"}:
            text = tool_call_text(name, label, raw_input).plain
            self.log.append(_LogEntry("tool-detail", f"  ⎿  {text.strip()}"))

    def _record_tool_result(self, tool_id: str, status: str, content: str) -> None:
        exploration = self._find_exploration_call(tool_id)
        if exploration is not None:
            exploration.complete = True
            exploration.failed = status != "ok"
            return
        name = self.tool_names.pop(tool_id, "")
        if status == "ok":
            detail = _successful_tool_detail(name, content)
            self.log.append(_LogEntry("tool-detail", f"  ⎿  {detail}"))
            return
        self.log.append(
            _LogEntry("tool-detail", f"  ⎿  ✗ {single_line_preview(content)}")
        )

    def _record_exploration_call(self, call: _ExplorationCall) -> None:
        if self.log and self.log[-1].role == "exploration":
            self.log[-1].exploration_calls.append(call)
            return
        self.log.append(_LogEntry("exploration", exploration_calls=[call]))

    def _find_exploration_call(self, tool_id: str) -> _ExplorationCall | None:
        for entry in reversed(self.log):
            if entry.role != "exploration":
                continue
            for call in reversed(entry.exploration_calls):
                if call.tool_id == tool_id:
                    return call
        return None

    def _handle_tool_update(self, tool_name: str, partial: str) -> None:
        if tool_name == "subagent":
            for line in partial.splitlines():
                self._record_subagent_update(line.strip())

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
        self._finish_thinking()

        final_answer = event.data.answer.strip()
        already_streamed = any(
            entry.role == "xcode" and entry.text.strip() == final_answer
            for entry in self.log
        )
        if final_answer and not already_streamed:
            self.log.append(_LogEntry("xcode", final_answer, markdown=True))

        reason = final_stop_reason(event.data)
        if reason:
            self.log.append(_LogEntry("stop", reason))
        self.thinking_core.reset()
        self.running = False

    def _finish_thinking(self) -> None:
        if self.thinking.strip():
            self.thinking_core.finish()
            text = self.thinking.strip()
            if self.thinking_duration_ms:
                text += (
                    f"\nThought for {format_elapsed(self.thinking_duration_ms / 1000)}"
                )
            self.log.append(_LogEntry("thinking", text))
        self.thinking_core.reset()


def _successful_tool_detail(name: str, content: str) -> str:
    """为成功工具调用生成紧凑详情，shell 保留可见输出。"""
    text = content.rstrip()
    if name in {"bash", "hypa_shell", "shell"}:
        return _tool_output_preview(text) if text else "done"
    if name == "read_file":
        count = sum(bool(re.match(r"^\d+: ", line)) for line in content.splitlines())
        return f"Read {count} lines"
    if name == "list_dir":
        entries = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith("...")
        ]
        if entries == ["(empty directory)"]:
            entries = []
        directories = sum(line.endswith("/") for line in entries)
        return f"{len(entries) - directories} files, {directories} directories"
    if name in {"glob_files", "find_files"}:
        return f"{len([line for line in content.splitlines() if line.strip()])} matches"
    if name == "grep_search":
        if "no matches" in content.lower():
            return "No matches found"
        matches = sum(bool(re.search(r":\d+:", line)) for line in content.splitlines())
        return f"{matches} matches"
    return "done"


def _is_exploration_call(name: str, raw_input: ToolInput) -> bool:
    """判断 agent 工具调用是否可安全归入只读探索组。"""
    if name in {"read_file", "list_dir", "grep_search", "glob_files", "find_files"}:
        return True
    if name != "bash":
        return False
    command = raw_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return False
    analysis = analyze_shell_command(command)
    return (
        analysis.ast_available
        and not analysis.parse_error
        and analysis.primary_command
        in {
            "cat",
            "head",
            "tail",
            "less",
            "more",
            "ls",
            "dir",
            "rg",
            "grep",
            "ack",
            "find",
        }
        and not analysis.unresolved_effects
        and all(path.access == "read" for path in analysis.resolved_paths)
    )


def _exploration_label(name: str, raw_input: ToolInput) -> str:
    """生成探索组中简洁且稳定的调用标题。"""
    if name == "bash":
        command = str(raw_input.get("command", ""))
        primary = analyze_shell_command(command).primary_command
        prefix = {
            "rg": "Search",
            "grep": "Search",
            "ack": "Search",
            "ls": "List",
            "dir": "List",
            "find": "List",
        }.get(primary, "Read")
        return f"{prefix} {single_line_preview(command)}"
    label = brief_input(name, raw_input)
    if name in {"grep_search", "grep", "rg", "ack"}:
        return f"Search {label.removeprefix('grep ')}"
    if name in {"glob_files", "find_files", "find", "list_dir", "ls", "dir"}:
        return f"List {label.removeprefix('glob ').removeprefix('ls ')}"
    if name in {"read_file", "read", "cat", "head", "tail", "less", "more"}:
        return f"Read {label.removeprefix('read ')}"
    return label[:1].upper() + label[1:]


def _tool_output_preview(content: str) -> str:
    """限制 shell 输出，避免单次命令占满 TUI。"""
    max_lines = 6
    max_chars = 800
    lines = content.splitlines()
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip()
    if len(lines) > max_lines:
        return f"{preview}\n  … {len(lines) - max_lines} lines omitted"
    if len(content) > len(preview):
        return f"{preview}\n  … output truncated"
    return preview
