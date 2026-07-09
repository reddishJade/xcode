"""轻量终端 TUI。"""

from __future__ import annotations

from contextlib import redirect_stdout
from collections.abc import Callable
from dataclasses import dataclass, field
from io import StringIO
import re
import textwrap
import threading
import time
from threading import Event
from typing import TYPE_CHECKING, cast

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.ansi import ANSI
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Label, TextArea
from rich.console import Console
from rich.markdown import Markdown

from xcode.agent.types import ToolInput, ToolSpec
from xcode.coding_agent.execution_modes import ExecutionMode
from xcode.harness.agent_runtime.events import (
    CodingAgentHarnessEvent,
    FinalStructuredEvent,
    ReasoningDeltaStructuredEvent,
    TextDeltaStructuredEvent,
    ToolResultStructuredEvent,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.harness.observability import HITLResult
from xcode.harness.observability.permission_model import (
    FileGrantStore,
    SessionGrantStoreManager,
)
from xcode.harness.session import SessionStore
from xcode.harness.snapshot import (
    SnapshotResult,
    SnapshotService,
    SnapshotStore,
    SnapshotUnsupportedError,
)

from .app_contract import ReplApp
from .commands import PromptText, ReplState
from .file_refs import expand_file_references
from .markdown import TerminalMarkdownRenderer
from .repl_commands import handle_command
from .repl_commands import COMMAND_NAMES
from .repl_hitl import parse_hitl_choice
from .repl_skills import activate_skill, parse_skill_invocation
from .repl_tools import (
    brief_input,
    event_to_dict,
    file_reference_event,
    final_stop_reason,
    run_shell_shortcut,
    tool_call_text,
)

if TYPE_CHECKING:
    from pathlib import Path

_file_ref_pattern = re.compile(r"(?<!\S)@([^\s]+)")


class _TuiInputLexer(Lexer):
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


class _TuiCompleter(Completer):
    """为 TUI 输入栏提供 / 命令补全。"""

    def __init__(self) -> None:
        self._commands = COMMAND_NAMES

    def get_completions(
        self, document: object, complete_event: object
    ) -> list[Completion]:
        text = str(getattr(document, "text", ""))
        if not text:
            return []
        if text.startswith("/"):
            partial = text.lower()
            return [
                Completion(cmd, start_position=-len(text))
                for cmd in self._commands
                if cmd.lower().startswith(partial)
            ]
        return []


class _TuiPromptSession:
    def prompt(self, prompt_text: PromptText) -> str:
        _ = prompt_text
        return ""


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
    log: list[_LogEntry] = field(default_factory=list)
    current_answer: str = ""
    thinking: str = ""
    thinking_start: float = 0.0
    thinking_duration_ms: int = 0
    tool_events: list[str] = field(default_factory=list)
    tool_labels: dict[str, _ToolSlot] = field(default_factory=dict)
    subagents: dict[int, _SubagentSlot] = field(default_factory=dict)
    pending_hitl: _HitlRequest | None = None
    thinking_collapsed: bool = False
    tool_collapsed: bool = False
    running: bool = False
    mode: ExecutionMode = "act"

    def add_user(self, text: str) -> None:
        self.log.append(_LogEntry("you", f"> {text}"))
        self.current_answer = ""
        self.thinking = ""
        self.thinking_start = 0.0
        self.thinking_duration_ms = 0
        self._clear_activity()

    def toggle_thinking(self) -> None:
        self.thinking_collapsed = not self.thinking_collapsed

    def toggle_tools(self) -> None:
        self.tool_collapsed = not self.tool_collapsed

    def handle_event(self, event: CodingAgentHarnessEvent) -> None:
        if isinstance(event, ReasoningDeltaStructuredEvent):
            if not self.thinking_start:
                self.thinking_start = time.time()
            self.thinking += event.data
        elif isinstance(event, TextDeltaStructuredEvent):
            self.current_answer += event.data
        elif isinstance(event, ToolUseStructuredEvent):
            self._record_tool_use(event.data.id, event.data.name, event.data.input)
        elif isinstance(event, ToolUpdateStructuredEvent):
            self._handle_tool_update(event.data.tool_name, event.data.partial_result)
        elif isinstance(event, ToolResultStructuredEvent):
            self._record_tool_result(
                event.data.tool_use_id,
                event.data.status,
                event.data.content,
            )
        elif isinstance(event, FinalStructuredEvent):
            self._finish_answer(event)

    def render(self) -> str:
        return "\n".join(self.lines()).rstrip() + "\n"

    def fragments(
        self, limit: int | None = None, scrollback: int = 0
    ) -> StyleAndTextTuples:
        # ansi_lines for scroll = same total lines as lines() but ANSI-colored
        all_ansi = self.ansi_lines()
        visible = _visible_lines(all_ansi, limit, scrollback)
        result: StyleAndTextTuples = []
        for line in visible:
            result.extend(_render_line_fragments(line))
            result.append(("", "\n"))
        return result

    def top_bar(self, project_name: str) -> str:
        state = "busy" if self.running else "idle"
        return (
            f" Xcode  ·  {state}  ·  mode {self.mode}  ·  cwd {project_name} "
        )

    def status(self, scrollback: int = 0) -> str:
        state = "busy" if self.running else "idle"
        thinking = "thinking:collapsed" if self.thinking_collapsed else "thinking:open"
        tools = "tools:collapsed" if self.tool_collapsed else "tools:open"
        scroll = f" · scroll {scrollback}" if scrollback else ""
        return f" {state} · Ctrl-T {thinking} · Ctrl-O {tools}{scroll} "

    def lines(self) -> list[str]:
        lines: list[str] = []
        for entry in self.log:
            if lines:
                lines.append("")
            if not entry.role:
                lines.extend(entry.text.splitlines())
            elif entry.role == "thinking":
                entry_lines = entry.text.splitlines() or [""]
                has_timing = any("Thought for" in ln for ln in entry_lines)
                dur_text = entry_lines[-1].strip() if has_timing else ""
                if has_timing:
                    entry_lines = entry_lines[:-1]
                bar = "│ thinking Thinking…"
                if dur_text:
                    bar += f" (Thought for {dur_text.split()[-1]})"
                lines.append(bar)
                if not self.thinking_collapsed:
                    for tl in entry_lines:
                        lines.append(f"│   {tl}")
            elif entry.markdown:
                lines.append(f"│ {entry.role}")
                for line in _rendered_markdown_lines(entry.text):
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
            lines.extend(_rendered_markdown_lines(self.current_answer.strip()))
        return lines

    def ansi_lines(self) -> list[str]:
        """Like lines() but markdown content uses ANSI codes for coloring."""
        lines: list[str] = []
        for entry in self.log:
            if lines:
                lines.append("")
            if not entry.role:
                lines.extend(entry.text.splitlines())
            elif entry.role == "thinking":
                entry_lines = entry.text.splitlines() or [""]
                has_timing = any("Thought for" in ln for ln in entry_lines)
                dur_text = entry_lines[-1].strip() if has_timing else ""
                if has_timing:
                    entry_lines = entry_lines[:-1]
                bar = "│ thinking Thinking…"
                if dur_text:
                    bar += f" (Thought for {dur_text.split()[-1]})"
                lines.append(bar)
                if not self.thinking_collapsed:
                    for tl in entry_lines:
                        lines.append(f"│   {tl.lstrip()}")
            elif entry.markdown:
                lines.append(f"│ {entry.role}")
                for line in _markdown_ansi_lines(entry.text):
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
            for line in _markdown_ansi_lines(self.current_answer.strip()):
                lines.append(f"│   {line}")
        return lines

    def _thinking_lines(self, lines: list[str]) -> None:
        if lines:
            lines.append("")
        dur = self.thinking_duration_ms
        if dur:
            bar = f"│ thinking Thinking… (Thought for {dur}ms)"
        else:
            bar = "│ thinking Thinking…"
        lines.append(bar)
        if not self.thinking_collapsed:
            for tl in self.thinking.splitlines():
                lines.append(f"│   {tl.lstrip()}")
            if dur:
                lines.append(f"│   (Thought for {dur}ms)")

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

    def _record_tool_use(self, tool_id: str, name: str, raw_input: ToolInput) -> None:
        label = brief_input(name, raw_input)
        text = tool_call_text(name, label, raw_input).plain
        self.tool_labels[tool_id] = _ToolSlot(name=name, label=label, text=text)
        self.tool_events.append(_tool_block(name, "running", f"正在调用工具: {label}"))
        if name in {"todowrite", "subagent"}:
            self.tool_events.append(_tool_block(name, "input", text.strip()))

    def _record_tool_result(self, tool_id: str, status: str, content: str) -> None:
        slot = self.tool_labels.get(tool_id)
        name = slot.name if slot else tool_id
        label = slot.label if slot else tool_id
        if status == "ok":
            summary = _tail_line(content)
            detail = f"✓ {label}" + (f" -> {summary}" if summary else "")
            self.tool_events.append(_tool_block(name, "success", detail))
            return
        self.tool_events.append(_tool_block(name, "error", f"✗ {label}: {content}"))

    def _handle_tool_update(self, tool_name: str, partial: str) -> None:
        if tool_name != "subagent":
            clean = _tail_line(partial)
            if clean:
                self.tool_events.append(_tool_block(tool_name, "update", clean))
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

    def _finish_answer(self, event: FinalStructuredEvent) -> None:
        answer = self.current_answer.strip() or event.data.answer.strip()

        # Persist thinking text before clearing
        if self.thinking.strip():
            if self.thinking_start:
                self.thinking_duration_ms = int((time.time() - self.thinking_start) * 1000)
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
        self.thinking = ""
        self.running = False

    def _clear_activity(self) -> None:
        self.tool_events.clear()
        self.tool_labels.clear()
        self.subagents.clear()


def run_tui(app: ReplApp, project_root: Path, sessions_dir: Path) -> int:
    """运行 TUI。"""
    return _XcodeTui(app, project_root, sessions_dir).run()


class _XcodeTui:
    def __init__(
        self,
        app: ReplApp,
        project_root: Path,
        sessions_dir: Path | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self._agent_app = app
        self._project_root = project_root
        self._store = SessionStore(
            sessions_dir or project_root / ".local" / "sessions",
            project_root=project_root,
        )
        self._repl_state = ReplState()
        self._snapshot_store: SnapshotStore | None = None
        try:
            self._snapshot_store = SnapshotStore(project_root)
        except SnapshotUnsupportedError:
            pass
        self._state = _TuiState(mode=self._repl_state.mode)
        self._scrollback = 0
        self._grant_store_manager = SessionGrantStoreManager()
        self._markdown_renderer = TerminalMarkdownRenderer()
        self._prompt_session = _TuiPromptSession()
        self._top = Label(text="", style="class:top")
        self._output_control = FormattedTextControl(
            self._fragments,
            focusable=False,
        )
        self._output = Window(
            self._output_control,
            wrap_lines=True,
            always_hide_cursor=True,
        )
        self._input = TextArea(
            height=3,
            prompt=self._input_prompt,
            multiline=False,
            completer=self._make_completer(),
            lexer=_TuiInputLexer(),
            complete_while_typing=True,
        )
        self._status = Label(text="", style="class:status")
        self._application = Application(
            layout=Layout(
                HSplit(
                    [
                        self._top,
                        self._output,
                        self._status,
                        Label(text="─" * 120, style="class:border"),
                        self._input,
                    ]
                )
            ),
            key_bindings=self._bindings(),
            full_screen=False,
            mouse_support=True,
            style=Style.from_dict(
                {
                    "": "",
                    "top": "ansigreen bold",
                    "user": "ansicyan bold",
                    "assistant": "ansigreen bold",
                    "thinking": "ansibrightblack",
                    "tool": "ansibrightblack",
                    "tool-title": "ansibrightblack bold",
                    "error": "ansired",
                    "status": "ansigreen",
                    "border": "ansibrightblack",
                }
            ),
            input=input,
            output=output,
        )
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            agent.approval_callback = self._approval_callback
            agent.session_id = self._store.session_id
            if hasattr(agent, "set_session_grant_store_provider"):
                agent.set_session_grant_store_provider(
                    lambda: self._grant_store_manager.get_for_session(
                        getattr(agent, "session_id", "tui")
                    )
                )
            if hasattr(agent, "set_permanent_grant_store"):
                agent.set_permanent_grant_store(
                    FileGrantStore.for_project_root(self._project_root)
                )
        self._application.layout.focus(self._input)
        self._refresh()

    def run(self) -> int:
        self._application.run()
        return 0

    def _make_completer(self) -> _TuiCompleter:
        return _TuiCompleter()

    def _input_prompt(self) -> str:
        return "授权 > " if self._state.pending_hitl is not None else "输入消息 > "

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        bindings.add("enter")(self._submit_key)
        bindings.add("pageup")(self._page_up_key)
        bindings.add("pagedown")(self._page_down_key)
        bindings.add("end")(self._end_key)
        bindings.add("c-t")(self._toggle_thinking_key)
        bindings.add("c-o")(self._toggle_tools_key)
        bindings.add(Keys.ScrollUp)(self._scroll_up_key)
        bindings.add(Keys.ScrollDown)(self._scroll_down_key)
        bindings.add("c-q")(self._quit_key)
        bindings.add("c-c")(self._cancel_key)
        return bindings

    def _submit_key(self, _event: object) -> None:
        text = self._input.text.strip()
        if not text:
            return
        self._input.text = ""
        self._scrollback = 0
        if self._state.pending_hitl is not None:
            self._answer_hitl_text(text)
            return
        if self._state.running:
            return
        self._submit(text)

    def _page_up_key(self, _event: object) -> None:
        self._scroll_by(10)

    def _page_down_key(self, _event: object) -> None:
        self._scroll_by(-10)

    def _scroll_up_key(self, _event: object) -> None:
        self._scroll_by(3)

    def _scroll_down_key(self, _event: object) -> None:
        self._scroll_by(-3)

    def _end_key(self, _event: object) -> None:
        self._scrollback = 0
        self._refresh()

    def _toggle_thinking_key(self, _event: object) -> None:
        self._state.toggle_thinking()
        self._repl_state.thinking_collapsed = self._state.thinking_collapsed
        self._refresh()

    def _toggle_tools_key(self, _event: object) -> None:
        self._state.toggle_tools()
        self._repl_state.tool_collapsed = self._state.tool_collapsed
        self._refresh()

    def _quit_key(self, _event: object) -> None:
        self._application.exit()

    def _cancel_key(self, _event: object) -> None:
        if self._state.running:
            agent = getattr(self._agent_app, "agent", None)
            if agent is not None:
                token = getattr(agent, "cancellation_token", None)
                if token is not None:
                    token.cancel("interrupted by user")
            self._store.append(
                "event", {"type": "interrupted", "data": "interrupted by user"}
            )
            self._state.log.append(_LogEntry("stop", "[interrupted]"))
            self._state.running = False
            self._refresh()
            return
        self._application.exit()

    def _submit(self, text: str) -> None:
        skill_invocation = parse_skill_invocation(text)
        if skill_invocation is not None:
            skill_name, remaining_text = skill_invocation
            activation = activate_skill(
                self._agent_app,
                self._store,
                skill_name,
                mode=self._repl_state.mode,
            )
            self._state.log.append(_LogEntry("system", activation.message))
            if activation.status not in {"activated", "already_active"}:
                self._refresh()
                return
            if not remaining_text:
                self._refresh()
                return
            text = remaining_text
        if text.startswith("/"):
            self._run_command(text)
            self._refresh()
            return
        if text.startswith("!"):
            self._run_shell_shortcut(text)
            self._refresh()
            return

        self._store.append("user", text)
        expanded_text, references = expand_file_references(text, self._project_root)
        if references:
            self._store.append("event", file_reference_event(references))
        self._state.add_user(text)
        self._state.running = True
        self._agent_app.agent.cancellation_token.reset()
        self._refresh()
        thread = threading.Thread(
            target=self._run_turn,
            args=(expanded_text,),
            daemon=True,
        )
        thread.start()

    def _run_command(self, text: str) -> None:
        output_buffer = StringIO()
        with redirect_stdout(output_buffer):
            should_exit = handle_command(
                text,
                self._store,
                self._agent_app,
                self._markdown_renderer,
                self._repl_state,
                self._prompt_session,
                self._grant_store_manager.get_for_session(self._store.session_id),
                FileGrantStore.for_project_root(self._project_root),
                static_policy=getattr(self._agent_app.agent, "permission_policy", None),
                restricted_dirs=getattr(self._agent_app.agent, "restricted_dirs", ()),
                snapshot_store=self._snapshot_store,
            )
        self._state.mode = self._repl_state.mode
        output = output_buffer.getvalue().strip()
        if output:
            self._state.log.append(_LogEntry("system", output, markdown=True))
        if should_exit:
            self._application.exit()

    def _run_shell_shortcut(self, text: str) -> None:
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            token = getattr(agent, "cancellation_token", None)
            if token is not None:
                token.reset()
        output = run_shell_shortcut(text, self._agent_app)
        self._store.append("event", {"type": "shell_shortcut", "data": text})
        self._store.append("event", {"type": "tool_result", "data": output})
        self._state.log.append(_LogEntry("shell", output, markdown=False))

    def _approval_callback(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        request = _HitlRequest(
            tool_name=tool.name,
            preview=_hitl_preview_lines(tool, action_input),
            event=Event(),
        )
        self._state.pending_hitl = request
        self._refresh()
        self._application.invalidate()
        import time
        time.sleep(0.05)  # yield to event loop so HITL prompt renders before block
        request.event.wait()
        result = request.result or HITLResult("deny", "once")
        if self._state.pending_hitl is request:
            self._state.pending_hitl = None
            self._refresh()
        return result

    def _answer_hitl_text(self, text: str) -> None:
        result = parse_hitl_choice(text)
        if result is None:
            self._state.log.append(
                _LogEntry(
                    "system",
                    "permission choice must be: Deny, Allow (once), "
                    "Allow this session, or Always allow",
                )
            )
            self._refresh()
            return
        self._answer_hitl(result)

    def _answer_hitl(self, result: HITLResult) -> None:
        request = self._state.pending_hitl
        if request is None:
            return
        request.result = result
        self._state.pending_hitl = None
        request.event.set()
        self._refresh()

    def _run_turn(self, text: str) -> None:
        snapshot = self._snapshot_store
        _snapshot_ctx: (
            tuple[str, SnapshotService, SnapshotResult] | None
        ) = None
        if snapshot is not None:
            _turn_id = snapshot.next_turn_id(self._store.session_id)
            _service = snapshot.service(self._store.session_id)
            _pre_result = _service.track()
            _snapshot_ctx = (_turn_id, _service, _pre_result)

        answer = ""
        tool_names: list[str] = []
        try:
            for event in self._agent_app.ask_stream(text, mode=self._repl_state.mode):
                if event.type not in {
                    "message_start",
                    "message_stop",
                    "reasoning_delta",
                    "text_delta",
                }:
                    self._store.append("event", event_to_dict(event))
                if isinstance(event, FinalStructuredEvent):
                    answer = event.data.answer
                if isinstance(event, ToolUseStructuredEvent):
                    tool_names.append(event.data.name)
                self._state.handle_event(event)
                self._refresh()
        except Exception as exc:
            self._state.log.append(_LogEntry("error", f"[error] {exc}"))
            self._state.running = False
            self._refresh()
            return

        if answer:
            self._store.append("assistant", answer)
            self._store.update_summary()

        # Scroll to bottom now that turn completed
        self._scrollback = 0
        self._refresh()

        if _snapshot_ctx is not None:
            turn_id, service, pre_result = _snapshot_ctx
            assert snapshot is not None  # guaranteed by _snapshot_ctx being set
            try:
                post_result = service.track()
                changes = service.diff(
                    pre_result.snapshot_id, post_result.snapshot_id
                )
                skipped_files = [
                    *pre_result.skipped_files,
                    *post_result.skipped_files,
                ]
                snapshot.record_turn(
                    session_id=self._store.session_id,
                    turn_id=turn_id,
                    pre_snapshot_id=pre_result.snapshot_id,
                    post_snapshot_id=post_result.snapshot_id,
                    changed_files=changes,
                    skipped_files=skipped_files,
                    tool_names=tool_names,
                )
            except Exception:
                pass

    def _fragments(self) -> StyleAndTextTuples:
        return self._state.fragments(self._output_height(), self._scrollback)

    def _output_height(self) -> int:
        return max(1, self._application.output.get_size().rows - 6)

    def _max_scrollback(self) -> int:
        return max(0, len(self._state.lines()) - self._output_height())

    def _scroll_by(self, amount: int) -> None:
        self._scrollback = max(
            0, min(self._max_scrollback(), self._scrollback + amount)
        )
        self._refresh()

    def _refresh(self) -> None:
        self._scrollback = min(self._scrollback, self._max_scrollback())
        self._top.text = self._state.top_bar(self._project_root.name)
        self._status.text = self._state.status(self._scrollback)
        self._application.invalidate()


def _hitl_preview_lines(tool: ToolSpec, action_input: ToolInput) -> list[str]:
    lines = [f"Tool: {tool.name}"]
    if tool.name == "edit_file":
        path = action_input.get("path", "")
        lines.append(f"File: {path}")
        if action_input.get("replace_all", False):
            lines.append("Replace all occurrences")
        old_text = str(action_input.get("old_text", ""))
        new_text = str(action_input.get("new_text", ""))
        if old_text:
            lines.append(f"- {old_text[:280].replace(chr(10), '¶ ')}")
        if new_text:
            lines.append(f"+ {new_text[:280].replace(chr(10), '¶ ')}")
    elif tool.name == "bash":
        command = str(action_input.get("command") or action_input.get("input", ""))
        lines.append(f"Command: {command[:500] or '(empty)'}")
        parts = command.strip().split()
        if parts:
            lines.append(f"Command type: {parts[0].lower()}")
    elif tool.name == "write_file":
        path = action_input.get("path", "")
        content = str(action_input.get("content", ""))
        lines.append(f"File: {path}")
        if content:
            lines.append(f"Content: {len(content.splitlines())} lines")
    elif tool.name == "read_file":
        lines.append(f"File: {action_input.get('path', '')}")
    elif tool.name in {"grep_search", "glob_files", "find_files"}:
        pattern = (
            action_input.get("pattern")
            or action_input.get("query")
            or action_input.get("path", "")
        )
        lines.append(f"Pattern: {str(pattern)[:200]}")
        if action_input.get("path") or action_input.get("include"):
            lines.append(
                f"Search in: {action_input.get('path') or action_input.get('include')}"
            )
    else:
        lines.append(f"Input: {brief_input(tool.name, action_input)}")
    return lines


def _visible_lines(lines: list[str], limit: int | None, scrollback: int) -> list[str]:
    if limit is None or len(lines) <= limit:
        return lines
    end = max(limit, len(lines) - scrollback)
    start = max(0, end - limit)
    return lines[start:end]



def _tool_block(name: str, status: str, text: str) -> str:
    lines = [f"│ tool {name} {status}"]
    lines.extend(f"│   {line}" for line in _wrap_lines(text))
    return "\n".join(lines)


def _rendered_markdown_lines(text: str) -> list[str]:
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


def _markdown_ansi_lines(text: str) -> list[str]:
    """Render markdown to ANSI-colored lines for fragment rendering."""
    buffer = StringIO()
    Console(
        file=buffer,
        width=112,
        force_terminal=True,
        color_system="truecolor",
    ).print(Markdown(text))
    raw = buffer.getvalue()
    # Strip trailing whitespace from each Rich-padded line
    lines = [line.rstrip() for line in raw.splitlines()]
    # Drop trailing empty lines
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _render_line_fragments(line: str) -> StyleAndTextTuples:
    """Convert a single text line to styled fragments.

    If the line contains ANSI escape codes, parse them into
    prompt_toolkit style fragments.  Otherwise use the simple
    prefix-based style.
    """
    if "\x1b[" in line:
        try:
            return cast(StyleAndTextTuples, to_formatted_text(ANSI(line)))
        except Exception:
            pass
    style = _line_style(line)
    return [(style, line)]


def _wrap_lines(text: str) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [text]:
        wrapped.extend(textwrap.wrap(paragraph, width=112) or [""])
    return wrapped


def _tail_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _line_style(line: str) -> str:
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
    if stripped.startswith("│   [") or stripped.startswith("│       "):
        return "class:tool"
    if stripped.startswith("│ error") or "✗" in stripped or "工具失败" in stripped:
        return "class:error"
    if stripped.startswith("│"):
        return ""  # default style (white) — most content lines
    return ""
