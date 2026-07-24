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
)
from xcode.harness.security.shell_analyzer import analyze_shell_command


_THINKING_ANSI = "\x1b[38;2;128;128;128m"
_ANSI_RESET = "\x1b[0m"

if TYPE_CHECKING:
    from xcode.harness.agent_runtime.events import (
        AgentHarnessEvent,
        FinalStructuredEvent,
    )
    from xcode.harness.security import HITLResult
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
class _CommandChoiceRequest:
    """TUI 内命令选择菜单的状态。"""

    choices: list[tuple[str, object]]
    on_select: Callable[[object], None]


@dataclass
class _CommandTextRequest:
    """TUI 内命令文本表单的状态。"""

    prompt: str
    on_submit: Callable[[str], None]
    on_cancel: Callable[[], None] | None = None


@dataclass
class _QuestionChoiceRequest:
    """TUI 内 question 工具的选择状态。"""

    prompt: str
    choices: list[tuple[str, str]]
    multiple: bool
    event: Event
    result: list[str] = field(default_factory=list)


@dataclass
class _ExplorationCall:
    tool_id: str
    name: str
    label: str
    solo_label: str
    complete: bool = False
    failed: bool = False
    detail: str = ""


@dataclass
class _LogEntry:
    role: str = "system"
    text: str = ""
    markdown: bool = False
    exploration_calls: list[_ExplorationCall] = field(default_factory=list)
    text_parts: list[str] | None = None
    _ansi_cache_key: tuple[object, ...] | None = field(default=None, repr=False)
    _ansi_lines: list[str] | None = field(default=None, repr=False)

    def content(self) -> str:
        """返回日志内容；流式回答使用分块存储以避免反复复制全文。"""
        if self.text_parts is not None:
            return "".join(self.text_parts)
        return self.text

    def freeze_content(self) -> None:
        """将完成的流式内容固化，后续读取不再重复拼接分块。"""
        if self.text_parts is None:
            return
        self.text = "".join(self.text_parts)
        self.text_parts = None

    def cached_ansi_lines(
        self,
        render_key: tuple[object, ...],
        render: Callable[[], list[str]],
    ) -> list[str]:
        """每条已完成消息只保留一份 ANSI 渲染缓存。"""
        exploration_key = tuple(
            (
                call.tool_id,
                call.name,
                call.label,
                call.solo_label,
                call.complete,
                call.failed,
                call.detail,
            )
            for call in self.exploration_calls
        )
        cache_key = (
            self.role,
            self.text,
            self.markdown,
            exploration_key,
            ("streaming", len(self.text_parts))
            if self.text_parts is not None
            else ("complete",),
            *render_key,
        )
        if self._ansi_cache_key != cache_key or self._ansi_lines is None:
            self._ansi_cache_key = cache_key
            self._ansi_lines = render()
        return self._ansi_lines


@dataclass(frozen=True)
class _DisplayBlock:
    """一个可独立缓存和裁剪的显示块。"""

    lines: list[str]
    leading_blank: bool = False

    @property
    def line_count(self) -> int:
        return len(self.lines) + int(self.leading_blank)

    def line_at(self, index: int) -> str:
        if self.leading_blank:
            if index == 0:
                return ""
            index -= 1
        return self.lines[index]


@dataclass
class _TuiState:
    """TUI 核心状态。

    tool use/text_delta 事件按到达顺序直接追加到 self.log，
    和 CLI 的 Live.update() 逐事件流式输出一致。
    """

    log: list[_LogEntry] = field(default_factory=list)
    streaming_answer: _LogEntry | None = None
    tool_names: dict[str, str] = field(default_factory=dict)
    project_root: Path | None = None
    thinking_core: ReasoningCore = field(default_factory=ReasoningCore)
    subagents: dict[int, _SubagentSlot] = field(default_factory=dict)
    pending_hitl: _HitlRequest | None = None
    pending_command_choice: _CommandChoiceRequest | None = None
    pending_command_text: _CommandTextRequest | None = None
    pending_question_choice: _QuestionChoiceRequest | None = None
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
        self._commit_streaming_answer()
        self.log.append(_LogEntry("you", text))
        self.thinking_core.reset()
        self.subagents.clear()

    def add_command(self, text: str) -> None:
        """将已执行的 TUI 命令显示为命令消息，但不打断当前回合。"""
        self.log.append(_LogEntry("command", text))

    def toggle_thinking(self) -> None:
        self.thinking_collapsed = not self.thinking_collapsed

    def toggle_tools(self) -> None:
        self.tool_collapsed = not self.tool_collapsed

    def restore_history(self, records: list[SessionEntry]) -> None:
        """将会话分支重建为与实时输出一致的 TUI 日志。"""
        self.log.clear()
        self.streaming_answer = None
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
                str(data.get("permission_notice") or ""),
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

    def handle_event(self, event: AgentHarnessEvent) -> None:
        if event.type == "reasoning_delta":
            self.thinking_core.handle_delta(event.data)
        elif event.type == "text_delta":
            self._handle_text_delta(event.data)
        elif event.type == "tool_use":
            self._finish_thinking()
            self._commit_streaming_answer()
            self._record_tool_use(event.data.id, event.data.name, event.data.input)
        elif event.type == "tool_update":
            self._handle_tool_update(event.data.tool_name, event.data.partial_result)
        elif event.type == "tool_result":
            self._record_tool_result(
                event.data.tool_use_id,
                event.data.status,
                event.data.content,
                event.data.permission_notice or "",
            )
        elif event.type == "final":
            self._finish_answer(event)

    def render(self) -> str:
        return "\n".join(self.lines()).rstrip() + "\n"

    # ── 片段生成 ──

    def fragments(
        self, limit: int | None = None, scrollback: int = 0
    ) -> StyleAndTextTuples:
        blocks = self._display_blocks(color=True)
        total = sum(block.line_count for block in blocks)
        if limit is None or total <= limit:
            start = 0
            end = total
        else:
            end = max(limit, total - scrollback)
            start = max(0, end - limit)
        result: StyleAndTextTuples = []
        offset = 0
        for block in blocks:
            block_end = offset + block.line_count
            if block_end > start and offset < end:
                local_start = max(0, start - offset)
                local_end = min(block.line_count, end - offset)
                for index in range(local_start, local_end):
                    line = block.line_at(index)
                    result.extend(render_line_fragments(line))
                    result.append(("", "\n"))
            offset = block_end
            if offset >= end:
                break
        return result

    def line_count(self) -> int:
        """返回当前显示行数，复用完成消息的单份 ANSI 缓存。"""
        return sum(block.line_count for block in self._display_blocks(color=True))

    # ── 文本渲染（纯文本：用于滚动高度计算） ──

    def lines(self) -> list[str]:
        return self._flatten_blocks(self._display_blocks(color=False))

    # ── ANSI 渲染（彩色：用于片段生成） ──

    def ansi_lines(self) -> list[str]:
        return self._flatten_blocks(self._display_blocks(color=True))

    @staticmethod
    def _flatten_blocks(blocks: list[_DisplayBlock]) -> list[str]:
        lines: list[str] = []
        for block in blocks:
            if block.leading_blank:
                lines.append("")
            lines.extend(block.lines)
        return lines

    def _display_blocks(self, color: bool) -> list[_DisplayBlock]:
        """生成轻量布局索引；完成消息的行内容由消息自身缓存。"""
        md_fn = markdown_ansi_lines if color else rendered_markdown_lines
        blocks: list[_DisplayBlock] = []
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
            show_tool_expand = self.tool_collapsed and index == latest_tool_index
            show_tool_collapse = not self.tool_collapsed and (
                index == latest_detail_index
                or (entry.role == "exploration" and index == latest_tool_index)
            )
            lines = self._entry_block_lines(
                entry,
                md_fn,
                color,
                show_tool_expand,
                show_tool_collapse,
            )
            if lines:
                blocks.append(
                    _DisplayBlock(
                        lines,
                        leading_blank=bool(blocks)
                        and entry.role not in {"tool", "tool-detail", "exploration"},
                    )
                )

        if self.streaming_answer is not None:
            lines = self._entry_block_lines(
                self.streaming_answer,
                md_fn,
                color,
                False,
                False,
            )
            if lines:
                blocks.append(_DisplayBlock(lines, leading_blank=bool(blocks)))

        if self.thinking.strip():
            lines = self._thinking_lines(color=color)
            blocks.append(_DisplayBlock(lines, leading_blank=bool(blocks)))
        self._append_optional_block(blocks, self._subagent_lines())
        self._append_optional_block(blocks, self._hitl_lines())
        return blocks

    @staticmethod
    def _append_optional_block(blocks: list[_DisplayBlock], lines: list[str]) -> None:
        if lines:
            blocks.append(_DisplayBlock(lines, leading_blank=bool(blocks)))

    def _entry_block_lines(
        self,
        entry: _LogEntry,
        md_fn: Callable[[str], list[str]],
        color: bool,
        show_tool_expand: bool,
        show_tool_collapse: bool,
    ) -> list[str]:
        def render() -> list[str]:
            lines: list[str] = []
            self._append_entry_lines(
                entry,
                lines,
                md_fn,
                color_thinking=color,
                show_tool_expand=show_tool_expand,
                show_tool_collapse=show_tool_collapse,
            )
            return lines

        if not color:
            return render()
        return entry.cached_ansi_lines(
            (
                self.tool_collapsed,
                self.thinking_collapsed,
                show_tool_expand,
                show_tool_collapse,
            ),
            render,
        )

    # ── 内部渲染方法 ──

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
        if entry.role == "command":
            cmd_lines = entry.content().splitlines() or [""]
            lines.append(f"> {cmd_lines[0]}")
            lines.extend(f"  {line}" for line in cmd_lines[1:])
        elif entry.role == "you":
            user_lines = entry.content().splitlines() or [""]
            lines.append(f"> {user_lines[0]}")
            lines.extend(f"  {line}" for line in user_lines[1:])
        elif entry.role == "tool":
            suffix = " (ctrl+o to expand)" if show_tool_expand else ""
            lines.append(f"{entry.text}{suffix}")
        elif entry.role == "exploration":
            if len(entry.exploration_calls) == 1:
                self._append_single_exploration(
                    entry.exploration_calls[0],
                    lines,
                    show_tool_expand,
                    show_tool_collapse,
                )
                return
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
            detail_lines = entry.content().splitlines() or [""]
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
            lines.extend(md_fn(_render_citations(entry.content())))
        else:
            lines.extend(entry.content().splitlines() or [""])

    def _append_single_exploration(
        self,
        call: _ExplorationCall,
        lines: list[str],
        show_tool_expand: bool,
        show_tool_collapse: bool,
    ) -> None:
        """单个探索调用沿用普通工具的标题和结果布局。"""
        suffix = (
            " (ctrl+o to expand)" if self.tool_collapsed and show_tool_expand else ""
        )
        lines.append(f"● {call.solo_label}{suffix}")
        if self.tool_collapsed or not call.complete:
            return
        detail = call.detail or ("failed" if call.failed else "done")
        detail_lines = detail.splitlines() or [""]
        lines.extend(f"  ⎿  {line}" for line in detail_lines)
        if show_tool_collapse:
            lines[-1] += " (ctrl+o to collapse)"

    def _thinking_lines(self, color: bool) -> list[str]:
        lines: list[str] = []
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
        return lines

    def _append_thinking_entry(
        self, entry: _LogEntry, lines: list[str], plain_text: bool, color: bool
    ) -> None:
        entry_lines = entry.content().splitlines() or [""]
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

    def _subagent_lines(self) -> list[str]:
        if not self.subagents or self.tool_collapsed:
            return []
        lines: list[str] = []
        lines.append("● Subagents")
        for index in sorted(self.subagents):
            slot = self.subagents[index]
            lines.append(f"  └ [{index}] {slot.task}")
            if slot.tool:
                lines.append(f"      {slot.tool}")
        return lines

    def _hitl_lines(self) -> list[str]:
        if self.pending_hitl is None:
            return []
        lines: list[str] = []
        lines.append("? Authorization request")
        for line in self.pending_hitl.preview:
            lines.append(f"  {line}")
        return lines

    # ── 工具事件记录（直接追加到 log） ──

    def _handle_text_delta(self, delta: str) -> None:
        """追加 text_delta 到 log 的最后一个 xcode 条目（或在流式期间追加新行）。"""
        self._finish_thinking()
        self._append_or_update_answer(delta)

    def _append_or_update_answer(self, delta: str) -> None:
        if self.streaming_answer is None:
            self.streaming_answer = _LogEntry(
                "xcode", markdown=True, text_parts=[delta]
            )
        else:
            self.streaming_answer.text_parts = self.streaming_answer.text_parts or []
            self.streaming_answer.text_parts.append(delta)

    def _commit_streaming_answer(self) -> None:
        if self.streaming_answer is None:
            return
        if self.streaming_answer.content():
            self.streaming_answer.freeze_content()
            self.log.append(self.streaming_answer)
        self.streaming_answer = None

    def _record_tool_use(self, tool_id: str, name: str, raw_input: ToolInput) -> None:
        if _is_exploration_call(name, raw_input):
            self._record_exploration_call(
                _ExplorationCall(
                    tool_id,
                    name,
                    _exploration_label(name, raw_input),
                    self._tool_label(name, raw_input),
                )
            )
            return
        self.tool_names[tool_id] = name
        label = self._tool_label(name, raw_input)

        self.log.append(_LogEntry("tool", f"● {label}"))
        if name in {"todowrite", "subagent"}:
            text = tool_call_text(name, label, raw_input).plain
            self.log.append(_LogEntry("tool-detail", f"  ⎿  {text.strip()}"))

    def _tool_label(self, name: str, raw_input: ToolInput) -> str:
        """生成普通工具卡片使用的标题。"""
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
        return label

    def _record_tool_result(
        self, tool_id: str, status: str, content: str, permission_notice: str = ""
    ) -> None:
        exploration = self._find_exploration_call(tool_id)
        if exploration is not None:
            exploration.complete = True
            exploration.failed = status != "ok"
            detail = (
                _successful_tool_detail(exploration.name, content)
                if status == "ok"
                else f"✗ {single_line_preview(content)}"
            )
            exploration.detail = _append_permission_notice(detail, permission_notice)
            return
        name = self.tool_names.pop(tool_id, "")
        if status == "ok":
            detail = _successful_tool_detail(name, content)
            detail = _append_permission_notice(detail, permission_notice)
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

        self._commit_streaming_answer()

        final_answer = event.data.answer.strip()
        already_streamed = any(
            entry.role == "xcode" and entry.content().strip() == final_answer
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


def _append_permission_notice(detail: str, permission_notice: str) -> str:
    """把自动授权来源附加到工具执行摘要。"""
    if not permission_notice:
        return detail
    return f"{detail} · {permission_notice}"


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
        analysis.classification_available
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
        }.get(primary or "", "Read")
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
