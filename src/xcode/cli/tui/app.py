"""TUI 应用入口：_XcodeTui 主类。"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import threading
import time
from threading import Event
from collections.abc import Callable
from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout import Float, FloatContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea

from ..app_contract import ReplApp
from ..commands import ReplState
from ..completion import ReplCompleter
from ..file_refs import expand_file_references
from ..markdown import TerminalMarkdownRenderer
from ..repl_commands import COMMAND_NAMES, COMMAND_REGISTRY_EXPORT, handle_command
from ..repl_hitl import parse_hitl_choice
from ..repl_skills import activate_skill, parse_skill_invocation
from ..repl_tools import (
    event_to_dict,
    file_reference_event,
    run_shell_shortcut,
)
from .rendering import (
    hitl_preview_lines,
)
from .state import _HitlRequest, _LogEntry, _TuiState
from xcode.harness.session import SessionStore
from xcode.harness.snapshot import SnapshotStore, SnapshotUnsupportedError
from .widgets import TuiInputLexer, TuiPromptSession

from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    ToolUseStructuredEvent,
)

if TYPE_CHECKING:
    from pathlib import Path

    from xcode.harness.snapshot import SnapshotService, SnapshotResult

    from xcode.agent.types import ToolInput, ToolSpec
    from xcode.harness.observability import HITLResult
    from xcode.harness.observability.permission_model import (
        SessionGrantStoreManager,
    )
    from xcode.harness.session import SessionStore


def run_tui(app: ReplApp, project_root: Path, sessions_dir: Path) -> int:
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
        self._store: SessionStore = SessionStore(
            sessions_dir or project_root / ".local" / "sessions",
            project_root=project_root,
        )
        self._repl_state = ReplState()
        self._snapshot_store = _init_snapshot_store(project_root)
        self._state = _TuiState(mode=self._repl_state.mode)
        self._scrollback = 0
        self._grant_store_manager: SessionGrantStoreManager | None = None
        try:
            from xcode.harness.observability.permission_model import (
                SessionGrantStoreManager,
            )
            self._grant_store_manager = SessionGrantStoreManager()
        except ImportError:
            pass
        self._markdown_renderer = TerminalMarkdownRenderer()
        self._prompt_session = TuiPromptSession()

        # ── UI 组件 ──
        self._top = Label(text="", style="class:top")
        self._output_control = FormattedTextControl(text="", focusable=False)
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
            lexer=TuiInputLexer(),
            complete_while_typing=True,
            accept_handler=lambda buf: self._submit_key(None) or True,
        )
        self._status = Label(text="", style="class:status")

        # ── Application ──
        from xcode.harness.observability.permission_model import FileGrantStore

        self._application = Application(
            layout=Layout(
                FloatContainer(
                    HSplit(
                        [
                            self._top,
                            self._output,
                            self._status,
                            Label(text="─" * 120, style="class:border"),
                            self._input,
                        ]
                    ),
                    floats=[
                        Float(
                            content=CompletionsMenu(max_height=8, scroll_offset=2),
                            xcursor=True,
                            ycursor=True,
                        ),
                    ],
                ),
            ),
            key_bindings=self._bindings(),
            full_screen=False,
            mouse_support=True,
            style=Style.from_dict({
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
                "completion-menu": "bg:default",
                "completion-menu.completion": "bg:default fg:default",
                "completion-menu.completion.current": (
                    "bg:default fg:default bold underline"
                ),
                "completion-menu.meta.completion": "bg:default fg:default",
                "completion-menu.meta.completion.current": (
                    "bg:default fg:default bold underline"
                ),
            }),
            input=input,
            output=output,
        )

        # ── Wire agent hooks ──
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            agent.approval_callback = self._approval_callback
            agent.session_id = self._store.session_id
            if hasattr(agent, "set_session_grant_store_provider"):
                agent.set_session_grant_store_provider(
                    lambda: self._grant_store_manager.get_for_session(
                        getattr(agent, "session_id", "tui")
                    )
                    if self._grant_store_manager is not None
                    else None
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

    # ── 辅助 ──

    def _make_completer(self) -> ReplCompleter:
        registry = tuple(getattr(self._agent_app, "registry", ()) or ())
        return ReplCompleter(
            project_root=self._project_root,
            registry=registry,
            command_names=COMMAND_NAMES,
            command_registry=COMMAND_REGISTRY_EXPORT,
        )

    def _input_prompt(self) -> str:
        return "授权 > " if self._state.pending_hitl is not None else "输入消息 > "

    # ── 键绑定 ──

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

    # ── 提交 ──

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
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            token = getattr(agent, "cancellation_token", None)
            if token is not None:
                token.reset()
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
                self._grant_store_manager.get_for_session(self._store.session_id)
                if self._grant_store_manager is not None
                else None,
                None,  # FileGrantStore X
                static_policy=getattr(self._agent_app.agent, "permission_policy", None),
                restricted_dirs=getattr(self._agent_app.agent, "restricted_dirs", ()),
                snapshot_store=self._snapshot_store,  # type: ignore[arg-type]
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

    # ── HITL ──

    def _approval_callback(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        from xcode.harness.observability import HITLResult

        request = _HitlRequest(
            tool_name=tool.name,
            preview=hitl_preview_lines(tool.name, action_input),
            event=Event(),
        )
        self._state.pending_hitl = request
        self._show_hitl_dialog(request, tool, action_input)
        self._refresh()
        self._application.invalidate()
        time.sleep(0.05)
        request.event.wait()
        result = request.result or HITLResult("deny", "once")
        if self._state.pending_hitl is request:
            self._state.pending_hitl = None
            self._dismiss_hitl_dialog()
            self._refresh()
        return result

    def _show_hitl_dialog(
        self, request: _HitlRequest, tool: ToolSpec, action_input: ToolInput
    ) -> None:
        from xcode.harness.observability.permissions import HITLDecision, HITLScope

        def _make_handler(decision: HITLDecision, scope: HITLScope) -> Callable[[], None]:
            def _on_click() -> None:
                from xcode.harness.observability import HITLResult

                request.result = HITLResult(decision, scope)
                request.event.set()

            return _on_click

        buttons = [
            Button("Allow (once)", handler=_make_handler("allow", "once")),
            Button("Allow this session", handler=_make_handler("allow", "session")),
            Button("Always allow", handler=_make_handler("allow", "permanent")),
            Button(" Deny ", handler=_make_handler("deny", "once")),
        ]
        lines = hitl_preview_lines(tool.name, action_input)
        preview_text = "\n".join(lines)
        dialog = Dialog(
            title=f"Authorization: {tool.name}",
            body=Label(text=preview_text),
            buttons=buttons,
        )
        container = self._application.layout.container
        if isinstance(container, FloatContainer):
            container.floats.append(Float(content=dialog))

    def _dismiss_hitl_dialog(self) -> None:
        container = self._application.layout.container
        if not isinstance(container, FloatContainer):
            return
        new_floats = [f for f in container.floats if not isinstance(f.content, Dialog)]
        container.floats[:] = new_floats

    def _answer_hitl_text(self, text: str) -> None:
        # 支持数字选择：1→Allow(once) 2→Allow this session 3→Always allow 4→Deny
        numeric = {"1": "allow (once)", "2": "allow this session", "3": "always allow", "4": "deny"}
        translated = numeric.get(text.strip(), text)
        result = parse_hitl_choice(translated)
        if result is None:
            self._state.log.append(
                _LogEntry(
                    "system",
                    "选择: [1] Allow (once)  [2] Allow this session  "
                    "[3] Always allow  [4] Deny",
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

    # ── Turn 执行 ──

    def _run_turn(self, text: str) -> None:
        snapshot = self._snapshot_store
        _snapshot_ctx = _enter_snapshot_ctx(snapshot, self._store.session_id)

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
                if isinstance(event, (FinalStructuredEvent,)):
                    answer = event.data.answer
                if isinstance(event, (ToolUseStructuredEvent,)):
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

        self._scrollback = 0
        self._refresh()

        _exit_snapshot_ctx(snapshot, _snapshot_ctx, self._store.session_id, tool_names)

    # ── 刷新 ──

    def _fragments(self):
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
        self._output_control.text = self._fragments()
        self._application._invalidated = False
        self._application.invalidate()


# ── 模块级工具函数 ──


def _init_snapshot_store(project_root: Path) -> object | None:
    try:
        return SnapshotStore(project_root)
    except SnapshotUnsupportedError:
        return None


def _enter_snapshot_ctx(
    snapshot_store: object | None, session_id: str
) -> tuple[str, SnapshotService, SnapshotResult] | None:
    if snapshot_store is None:
        return None
    try:
        from typing import cast as _cast
        from xcode.harness.snapshot import SnapshotStore as _S
        store = _cast(_S, snapshot_store)
        turn_id = store.next_turn_id(session_id)
        service = store.service(session_id)
        pre_result = service.track()
        return (turn_id, service, pre_result)
    except Exception:
        return None


def _exit_snapshot_ctx(
    snapshot_store: object | None,
    ctx: tuple[str, SnapshotService, SnapshotResult] | None,
    session_id: str,
    tool_names: list[str],
) -> None:
    if snapshot_store is None or ctx is None:
        return
    try:
        from typing import cast as _cast
        from xcode.harness.snapshot import SnapshotStore as _S
        store = _cast(_S, snapshot_store)
        turn_id, service, pre_result = ctx
        post_result = service.track()
        changes = service.diff(
            pre_result.snapshot_id, post_result.snapshot_id
        )
        skipped_files = [
            *pre_result.skipped_files,
            *post_result.skipped_files,
        ]
        store.record_turn(
            session_id=session_id,
            turn_id=turn_id,
            pre_snapshot_id=pre_result.snapshot_id,
            post_snapshot_id=post_result.snapshot_id,
            changed_files=changes,
            skipped_files=skipped_files,
            tool_names=tool_names,
        )
    except Exception:
        pass
