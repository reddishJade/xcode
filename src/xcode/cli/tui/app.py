"""TUI 应用入口：_XcodeTui 主类。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING, cast

from prompt_toolkit.application import Application
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout import Float, FloatContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output.base import Output
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import RadioList, TextArea

from ..app_contract import ReplApp
from ..commands import ReplState
from ..completion import ReplCompleter
from ..file_refs import expand_file_references
from ..markdown import TerminalMarkdownRenderer
from ..repl_commands import COMMAND_NAMES, COMMAND_REGISTRY_EXPORT, handle_command
from ..repl_hitl import HITL_CHOICES, parse_hitl_choice
from ..repl_skills import activate_skill, parse_skill_invocation
from ..repl_tools import (
    brief_input,
    event_to_dict,
    file_reference_event,
    run_shell_shortcut,
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
        self._state = _TuiState(mode=self._repl_state.mode, project_root=project_root)
        self._scrollback = 0
        self._committing = False
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
        self._awaiting_denial_suggestion = False

        # ── UI 组件 ──
        self._output_control = FormattedTextControl(text="", focusable=False)
        self._output = Window(
            self._output_control,
            wrap_lines=True,
            always_hide_cursor=True,
            dont_extend_height=True,
        )
        self._input = TextArea(
            height=1,
            prompt=self._input_prompt,
            multiline=False,
            completer=self._make_completer(),
            lexer=TuiInputLexer(),
            complete_while_typing=True,
            accept_handler=lambda buf: self._submit_key(None) or True,
        )
        self._approval_choices = RadioList(
            [(choice, choice) for choice in HITL_CHOICES],
            default=HITL_CHOICES[0],
            show_numbers=True,
            select_on_focus=True,
            open_character="",
            select_character="❯",
            close_character="",
            show_cursor=False,
            show_scrollbar=False,
        )
        approval_bindings = cast(
            KeyBindings, self._approval_choices.control.key_bindings
        )
        approval_bindings.add("enter")(lambda _event: self._accept_approval_choice())
        self._approval_container = ConditionalContainer(
            self._approval_choices,
            filter=Condition(
                lambda: (
                    self._state.pending_hitl is not None
                    and not self._awaiting_denial_suggestion
                )
            ),
        )
        self._input_container = ConditionalContainer(
            self._input,
            filter=Condition(
                lambda: (
                    self._state.pending_hitl is None or self._awaiting_denial_suggestion
                )
            ),
        )
        # ── Application ──
        from xcode.harness.observability.permission_model import FileGrantStore

        self._application = Application(
            layout=Layout(
                FloatContainer(
                    HSplit(
                        [
                            self._output,
                            self._approval_container,
                            self._input_container,
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
            mouse_support=False,
            style=Style.from_dict(
                {
                    "": "",
                    "user": "ansicyan bold",
                    "thinking": "ansibrightblack",
                    "tool": "ansibrightblack",
                    "tool-title": "ansigreen bold",
                    "error": "ansired",
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
                    "radio-selected": "ansicyan bold",
                }
            ),
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
                    lambda: (
                        self._grant_store_manager.get_for_session(
                            getattr(agent, "session_id", "tui")
                        )
                        if self._grant_store_manager is not None
                        else None
                    )
                )
            if hasattr(agent, "set_permanent_grant_store"):
                agent.set_permanent_grant_store(
                    FileGrantStore.for_project_root(self._project_root)
                )

        self._application.layout.focus(self._input)
        self._refresh()

    def run(self) -> int:
        info = self._agent_app.get_model_info()
        print("Xcode")
        print(f"model: {info.get('model', 'unknown')}")
        print(f"mode:  {self._repl_state.mode}")
        print(f"cwd:   {self._project_root}")
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
        if self._awaiting_denial_suggestion:
            return "Tell model what to do > "
        return "> "

    # ── 键绑定 ──

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        bindings.add("enter")(self._submit_key)
        bindings.add("pageup")(self._page_up_key)
        bindings.add("pagedown")(self._page_down_key)
        bindings.add("end")(self._end_key)
        # 输入框带有 Emacs 风格的默认按键。这里必须抢占 Ctrl+T/Ctrl+O，
        # 否则快捷键会被输入控件消费，折叠状态不会及时重绘。
        bindings.add("c-t", eager=True)(self._toggle_thinking_key)
        bindings.add("c-o", eager=True)(self._toggle_tools_key)
        bindings.add(Keys.ScrollUp)(self._scroll_up_key)
        bindings.add(Keys.ScrollDown)(self._scroll_down_key)
        bindings.add("c-q")(self._quit_key)
        bindings.add("c-c")(self._cancel_key)
        return bindings

    def _submit_key(self, _event: object) -> None:
        text = self._input.text.strip()
        if self._awaiting_denial_suggestion:
            self._input.text = ""
            self._finish_denial(text)
            return
        if not text:
            return
        self._input.text = ""
        self._scrollback = 0
        if self._state.running or self._committing:
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
        self._update_preserving_viewport(self._state.toggle_thinking)
        self._repl_state.thinking_collapsed = self._state.thinking_collapsed
        self._refresh()

    def _toggle_tools_key(self, _event: object) -> None:
        self._update_preserving_viewport(self._state.toggle_tools)
        self._repl_state.tool_collapsed = self._state.tool_collapsed
        self._refresh()

    def _quit_key(self, _event: object) -> None:
        if self._state.pending_hitl is not None:
            self._finish_denial("")
        self._application.exit()

    def _cancel_key(self, _event: object) -> None:
        if self._state.pending_hitl is not None:
            self._finish_denial("")
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
        self._state.running = True

        def invoke() -> bool:
            return handle_command(
                text,
                self._store,
                self._agent_app,
                self._markdown_renderer,
                self._repl_state,
                self._prompt_session,
                self._grant_store_manager.get_for_session(self._store.session_id)
                if self._grant_store_manager is not None
                else None,
                None,
                static_policy=getattr(self._agent_app.agent, "permission_policy", None),
                restricted_dirs=getattr(self._agent_app.agent, "restricted_dirs", ()),
                snapshot_store=cast(SnapshotStore | None, self._snapshot_store),
            )

        async def run() -> None:
            should_exit = await run_in_terminal(invoke, in_executor=True)
            self._state.mode = self._repl_state.mode
            self._state.running = False
            self._refresh()
            if should_exit:
                self._application.exit()

        if self._application.loop is None:
            asyncio.run(run())
        else:
            self._application.create_background_task(run())

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
            preview=[
                f"Authorization required: {tool.name}",
                f"Input: {brief_input(tool.name, action_input)}",
            ],
            event=Event(),
        )
        self._state.pending_hitl = request

        def show_choices() -> None:
            self._approval_choices._selected_index = 0
            self._approval_choices.current_value = HITL_CHOICES[0]
            self._application.layout.focus(self._approval_choices)
            self._refresh()

        loop = self._application.loop
        if loop is None:
            show_choices()
        else:
            loop.call_soon_threadsafe(show_choices)
        request.event.wait()
        result = request.result or HITLResult("deny", "once")
        if self._state.pending_hitl is request:
            self._state.pending_hitl = None
        return result

    def _accept_approval_choice(self) -> None:
        choice = self._approval_choices.current_value
        if choice == "Deny":
            self._awaiting_denial_suggestion = True
            self._application.layout.focus(self._input)
            self._refresh()
            return
        result = parse_hitl_choice(choice)
        if result is not None:
            self._complete_approval(result)

    def _finish_denial(self, suggestion: str) -> None:
        from xcode.harness.observability import HITLResult

        self._complete_approval(HITLResult("deny", "once", suggestion=suggestion))

    def _complete_approval(self, result: HITLResult) -> None:
        request = self._state.pending_hitl
        if request is None:
            return
        request.result = result
        self._state.pending_hitl = None
        self._awaiting_denial_suggestion = False
        request.event.set()
        self._application.layout.focus(self._input)
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
            self._schedule_turn_commit()
            return

        if answer:
            self._store.append("assistant", answer)
            self._store.update_summary()

        self._scrollback = 0
        self._refresh()

        _exit_snapshot_ctx(snapshot, _snapshot_ctx, self._store.session_id, tool_names)
        self._schedule_turn_commit()

    def _schedule_turn_commit(self) -> None:
        loop = self._application.loop
        if loop is None:
            return
        self._committing = True
        loop.call_soon_threadsafe(
            lambda: self._application.create_background_task(self._commit_turn())
        )

    async def _commit_turn(self) -> None:
        transcript = FormattedText(self._state.fragments())
        await run_in_terminal(
            lambda: print_formatted_text(
                transcript,
                end="",
                style=self._application.style,
                output=self._application.output,
            )
        )
        self._state.log.clear()
        self._state.tool_names.clear()
        self._state.subagents.clear()
        self._scrollback = 0
        self._committing = False
        self._refresh()

    # ── 刷新 ──

    def _fragments(self):
        return self._state.fragments(self._output_height(), self._scrollback)

    def _output_height(self) -> int:
        approval_height = (
            len(HITL_CHOICES)
            if self._state.pending_hitl is not None
            and not self._awaiting_denial_suggestion
            else 0
        )
        return max(1, self._application.output.get_size().rows - 1 - approval_height)

    def _max_scrollback(self) -> int:
        return max(0, len(self._state.lines()) - self._output_height())

    def _scroll_by(self, amount: int) -> None:
        self._scrollback = max(
            0, min(self._max_scrollback(), self._scrollback + amount)
        )
        self._refresh()

    def _update_preserving_viewport(self, update: Callable[[], None]) -> None:
        """更新会改变行数的显示状态，并保持当前视口的顶部位置。"""
        top_line = max(
            0,
            len(self._state.lines()) - self._output_height() - self._scrollback,
        )
        update()
        self._scrollback = max(
            0,
            len(self._state.lines()) - self._output_height() - top_line,
        )

    def _refresh(self) -> None:
        self._scrollback = min(self._scrollback, self._max_scrollback())
        self._output_control.text = self._fragments()
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
        changes = service.diff(pre_result.snapshot_id, post_result.snapshot_id)
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
