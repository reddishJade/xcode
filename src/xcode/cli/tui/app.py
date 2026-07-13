"""TUI 应用入口：_XcodeTui 主类。"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from io import StringIO
import re
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, cast

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout import Float, FloatContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import RadioList, TextArea

from ..app_contract import ReplApp
from ..commands import ReplState
from ..completion import CommandArgsSuggester, ReplCompleter
from ..file_refs import expand_file_references
from ..markdown import TerminalMarkdownRenderer
from ..repl_commands import COMMAND_NAMES, COMMAND_REGISTRY_EXPORT, handle_command
from ..repl_hitl import HITL_CHOICES, parse_hitl_choice, tool_preview_lines
from ..repl import current_effort_options, current_model_options
from ..repl_sessions import (
    print_saved_conversation,
    select_session_interactively,
    sync_agent_history,
)
from ..repl_skills import activate_skill, available_skill_names, parse_skill_invocation
from ..repl_tools import (
    event_to_dict,
    file_reference_event,
    run_shell_shortcut,
)
from xcode.agent.messages import UserMessage
from xcode.harness.agent_runtime import SubmitStatus
from .state import (
    _CommandChoiceRequest,
    _CommandTextRequest,
    _HitlRequest,
    _LogEntry,
    _TuiState,
)
from xcode.harness.session import SessionStore
from xcode.harness.snapshot import SnapshotStore, SnapshotUnsupportedError
from .widgets import TuiInputLexer, TuiPromptSession, tui_input_prompt

from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    ToolUseStructuredEvent,
)

_SHORTCUT_HELP = """Shortcuts
  ?              show this help
  Ctrl+C         interrupt; press twice to exit when idle
  Ctrl+Q         exit
  Ctrl+T         expand or collapse thinking
  Ctrl+O         expand or collapse tool details
  ! command      run a bash command
  $skill [task]  invoke a skill
  /command       run a slash command
  Tab            complete commands, skills, and @file references
  Shift+Enter    insert a newline (Esc Enter also works)
  PageUp/Down    scroll history; End returns to the latest output"""

if TYPE_CHECKING:
    from prompt_toolkit.history import History

    from xcode.harness.snapshot import SnapshotService, SnapshotResult

    from xcode.agent.types import ToolInput, ToolSpec
    from xcode.harness.observability import HITLResult
    from xcode.harness.observability.permission_model import (
        SessionGrantStoreManager,
    )
    from xcode.harness.session import SessionStore


def run_tui(
    app: ReplApp,
    project_root: Path,
    sessions_dir: Path,
    *,
    resume_latest: bool = False,
    auto_continue: bool = False,
    session_id: str | None = None,
) -> int:
    try:
        return _XcodeTui(
            app,
            project_root,
            sessions_dir,
            resume_latest=resume_latest,
            auto_continue=auto_continue,
            session_id=session_id,
        ).run()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1


class _XcodeTui:
    def __init__(
        self,
        app: ReplApp,
        project_root: Path,
        sessions_dir: Path | None = None,
        input: Input | None = None,
        output: Output | None = None,
        *,
        resume_latest: bool = False,
        auto_continue: bool = False,
        session_id: str | None = None,
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
        from xcode.harness.observability.permission_model import FileGrantStore

        self._permanent_grant_store = FileGrantStore.for_project_root(project_root)
        self._restore_startup_session(resume_latest, auto_continue, session_id)
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
        self._exit_pending = 0.0

        # ── UI 组件 ──
        self._output_control = FormattedTextControl(text="", focusable=False)
        self._output = Window(
            self._output_control,
            wrap_lines=True,
            always_hide_cursor=True,
            dont_extend_height=True,
        )
        completer = self._make_completer()
        self._input = TextArea(
            height=lambda: Dimension.exact(self._input_height()),
            prompt=self._input_prompt,
            multiline=True,
            completer=completer,
            lexer=TuiInputLexer(),
            complete_while_typing=True,
            auto_suggest=CommandArgsSuggester(completer.command_args),
            history=_tui_history(project_root),
        )
        self._input.buffer.on_text_insert += lambda _buf: setattr(
            self._input.buffer, "complete_state", None
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
        self._command_choices: RadioList[object] = RadioList(
            [(None, "")],
            show_numbers=True,
            select_on_focus=True,
            open_character="",
            select_character="❯",
            close_character="",
            show_cursor=False,
            show_scrollbar=False,
        )
        command_bindings = cast(KeyBindings, self._command_choices.control.key_bindings)
        command_bindings.add("enter")(lambda _event: self._accept_command_choice())
        self._approval_container = ConditionalContainer(
            self._approval_choices,
            filter=Condition(
                lambda: (
                    self._state.pending_hitl is not None
                    and not self._awaiting_denial_suggestion
                )
            ),
        )
        self._command_container = ConditionalContainer(
            self._command_choices,
            filter=Condition(lambda: self._state.pending_command_choice is not None),
        )
        input_visible = Condition(
            lambda: (
                self._state.pending_command_choice is None
                and (
                    self._state.pending_hitl is None or self._awaiting_denial_suggestion
                )
            )
        )
        self._input_container = ConditionalContainer(
            HSplit(
                [
                    Window(height=1, char="─", style="class:input-border"),
                    self._input,
                    Window(height=1, char="─", style="class:input-border"),
                ]
            ),
            filter=input_visible,
        )
        self._status = Window(
            FormattedTextControl(text=self._status_text),
            height=1,
            style="class:status",
        )
        # ── Application ──
        self._application = Application(
            layout=Layout(
                FloatContainer(
                    HSplit(
                        [
                            self._output,
                            self._approval_container,
                            self._command_container,
                            self._input_container,
                            self._status,
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
                    "thinking": "#808080",
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
                    "status": "ansibrightblack",
                    "input-border": "ansibrightblack",
                    "prompt-marker": "ansiyellow bold",
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
                agent.set_permanent_grant_store(self._permanent_grant_store)
            from ..repl_commands import _compute_context_summary

            _compute_context_summary(agent, self._project_root, self._repl_state)

        self._state.log.append(_LogEntry("system", self._header_text()))
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
            effort_options=lambda: current_effort_options(self._agent_app),
            model_options=lambda: current_model_options(self._agent_app),
            skill_options=lambda: available_skill_names(self._agent_app),
        )

    def _input_prompt(self) -> FormattedText:
        request = self._state.pending_command_text
        if request is not None:
            return FormattedText([("", f"{request.prompt} > ")])
        return FormattedText(
            tui_input_prompt(
                self._awaiting_denial_suggestion,
                self._input.text.startswith("!"),
            )
        )

    # ── 键绑定 ──

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        bindings.add("enter")(self._submit_key)
        try:
            bindings.add("s-enter")(self._insert_newline)
        except ValueError:
            pass
        bindings.add("escape", "enter")(self._insert_newline)
        bindings.add("pageup")(self._page_up_key)
        bindings.add("pagedown")(self._page_down_key)
        bindings.add("end")(self._end_key)
        bindings.add(Keys.ScrollUp)(self._scroll_up_key)
        bindings.add(Keys.ScrollDown)(self._scroll_down_key)
        bindings.add("?")(self._show_shortcuts_key)
        # 折叠快捷键必须是应用级绑定：焦点可能在授权列表等非输入控件。
        bindings.add("c-t", eager=True)(self._toggle_thinking_key)
        bindings.add("c-o", eager=True)(self._toggle_tools_key)
        bindings.add("c-q")(self._quit_key)
        bindings.add("c-c")(self._cancel_key)
        return bindings

    def _submit_key(self, _event: object) -> None:
        text = self._input.text.strip()
        if self._state.pending_command_text is not None:
            self._input.text = ""
            self._submit_command_text(text)
            return
        if self._awaiting_denial_suggestion:
            self._input.text = ""
            self._finish_denial(text)
            return
        if not text:
            return
        self._input.text = ""
        self._scrollback = 0
        if self._state.running or self._committing:
            if self._state.running and _is_live_command(text):
                self._run_command(text, preserve_running=True)
            elif self._state.running and not text.startswith(("/", "!", "$")):
                self._submit_busy_message(text)
            return
        self._submit(text)

    def _submit_busy_message(self, text: str) -> None:
        """将忙时普通输入按默认 steer policy 交给 session controller。"""
        expanded_text, references = expand_file_references(text, self._project_root)
        outcome = self._agent_app.agent.submit_busy_message(
            UserMessage(content=expanded_text),
            self._repl_state.busy_mode,
        )
        self._state.add_user(text)
        if references:
            self._store.append("event", file_reference_event(references))
        if outcome.status is SubmitStatus.STEER_ACCEPTED:
            self._store.append("user", text)
            self._state.log.append(
                _LogEntry("system", f"[steer] accepted by {outcome.run_id}")
            )
        elif outcome.status is SubmitStatus.FOLLOW_UP_QUEUED:
            self._state.log.append(
                _LogEntry(
                    "system",
                    f"[{self._repl_state.busy_mode.value}] queued for the next run",
                )
            )
        elif outcome.status is SubmitStatus.INTERRUPT_REQUESTED:
            self._state.log.append(
                _LogEntry("system", "[interrupt] cancelling before replacement run")
            )
        else:
            self._repl_state.pending_inject = expanded_text
            self._state.log.append(
                _LogEntry("system", "[followup] active run already finished")
            )
        self._refresh()

    def _insert_newline(self, event: object) -> None:
        buffer = getattr(event, "current_buffer", None)
        if buffer is not None:
            buffer.insert_text("\n")

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

    def _show_shortcuts_key(self, event: object) -> None:
        if self._input.text:
            buffer = getattr(event, "current_buffer", None)
            if buffer is not None:
                buffer.insert_text("?")
            return
        self._state.log.append(_LogEntry("system", _SHORTCUT_HELP))
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
        print_saved_conversation(self._store)
        self._application.exit()

    def _cancel_key(self, _event: object) -> None:
        if self._state.pending_command_choice is not None:
            self._state.pending_command_choice = None
            self._application.layout.focus(self._input)
            self._refresh()
            return
        if self._state.pending_command_text is not None:
            self._state.pending_command_text = None
            self._refresh()
            return
        if self._state.pending_hitl is not None:
            self._finish_denial("")
        if self._state.running:
            agent = getattr(self._agent_app, "agent", None)
            if agent is not None:
                interrupt = getattr(agent, "interrupt", None)
                if callable(interrupt):
                    interrupt("interrupted by user")
                else:
                    token = getattr(agent, "cancellation_token", None)
                    if token is not None:
                        token.cancel("interrupted by user")
            self._store.append(
                "event", {"type": "interrupted", "data": "interrupted by user"}
            )
            self._state.log.append(_LogEntry("stop", "[interrupt requested]"))
            self._refresh()
            return
        now = perf_counter()
        if self._exit_pending and now - self._exit_pending < 1.5:
            print_saved_conversation(self._store)
            self._application.exit()
            return
        self._exit_pending = now
        self._state.log.append(_LogEntry("system", "(press Ctrl+C again to exit)"))
        self._refresh()

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

    def _run_command(self, text: str, *, preserve_running: bool = False) -> None:
        """执行斜杠命令；实时命令不得结束正在运行的 agent 回合。"""
        if text in {"/clear", "/new"}:
            self._clear_session()
            return
        if self._show_native_command_choice(text):
            return

        if not preserve_running:
            self._state.running = True

        def invoke(capture_output: bool) -> tuple[bool, str]:
            def handle() -> bool:
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
                    self._permanent_grant_store,
                    static_policy=getattr(
                        self._agent_app.agent, "permission_policy", None
                    ),
                    restricted_dirs=getattr(
                        self._agent_app.agent, "restricted_dirs", ()
                    ),
                    snapshot_store=cast(SnapshotStore | None, self._snapshot_store),
                    show_session_history=True,
                )

            if not capture_output:
                return handle(), ""
            output = StringIO()
            with redirect_stdout(output):
                should_exit = handle()
            return should_exit, output.getvalue()

        async def run_inline() -> None:
            should_exit, output = await asyncio.get_running_loop().run_in_executor(
                None, invoke, True
            )
            if output.strip():
                self._state.log.append(_LogEntry("system", output.rstrip()))
            if preserve_running:
                self._refresh()
                if (
                    self._repl_state.pending_inject is not None
                    and not self._state.running
                ):
                    self._submit_pending_inject()
            else:
                self._finish_command(text, should_exit)

        if self._application.loop is None:
            asyncio.run(run_inline())
        else:
            self._application.create_background_task(run_inline())

    def _show_native_command_choice(self, text: str) -> bool:
        """为需要选择的会话命令打开 TUI 原生菜单。"""
        command = text.split(maxsplit=1)[0]
        if command == "/permissions":
            self._open_command_choices(
                [
                    ("Show permission status", "/permissions list"),
                    ("Clear session permissions", "/permissions clear"),
                ],
                lambda selected: self._run_command(str(selected)),
            )
            return True
        if command == "/config":
            parts = text.split(maxsplit=2)
            action = parts[1] if len(parts) > 1 else ""
            if action == "add":
                self._state.log.append(
                    _LogEntry(
                        "system",
                        "Adding a provider profile requires a multi-step TUI form; "
                        "that form is not available yet.",
                    )
                )
                self._refresh()
                return True
            if action in {"edit", "set"} and len(parts) < 3:
                self._open_config_profile_editor()
                return True
            if action == "delete":
                self._open_config_profile_delete()
                return True
        if command == "/fork":
            entries = self._store.get_forkable_user_messages()
            if not entries:
                self._state.log.append(
                    _LogEntry("system", "No user messages to fork from.")
                )
                self._refresh()
                return True

            def fork(entry: object) -> None:
                parent_session_id = self._store.session_id
                entry_id = getattr(entry, "id", "")
                try:
                    forked = self._store.fork_from_entry(entry_id)
                except ValueError as exc:
                    self._state.log.append(_LogEntry("error", f"[error] {exc}"))
                    return
                self._store.current_path = forked.current_path
                meta = self._store.current_metadata()
                if self._snapshot_store is not None and meta is not None:
                    cast(SnapshotStore, self._snapshot_store).fork_session(
                        parent_session_id, meta.id
                    )
                sync_agent_history(self._agent_app, self._store)
                self._state.restore_history(self._store.build_branch())
                self._state.log.append(
                    _LogEntry("system", f'Forked at: "{meta.title if meta else ""}"')
                )

            self._open_command_choices(
                [(" ".join(str(e.content).split())[:100], e) for e in entries], fork
            )
            return True
        if command in {"/sessions", "/resume"} and len(text.split()) == 1:
            sessions = self._store.list_infos()
            if not sessions:
                self._state.log.append(_LogEntry("system", "No conversations found."))
                self._refresh()
                return True

            def resume(session: object) -> None:
                self._store.resume(getattr(session, "id", ""))
                sync_agent_history(self._agent_app, self._store)
                self._restore_session_history()

            self._open_command_choices(
                [(f"{s.title} ({s.id[:8]})", s) for s in sessions], resume
            )
            return True
        if command == "/tree":
            nodes = self._store.get_tree()
            if not nodes:
                self._state.log.append(
                    _LogEntry("system", "No session tree available (no metadata).")
                )
                self._refresh()
                return True

            def jump(node: object) -> None:
                node_id = getattr(node, "id", "")
                if not self._store.jump_to_entry(node_id):
                    self._state.log.append(_LogEntry("error", "Failed to set entry."))
                    return
                sync_agent_history(self._agent_app, self._store)
                self._state.restore_history(self._store.build_branch())

            self._open_command_choices(
                [
                    (
                        f"{'  ' * n.depth}{'└─ ' if n.depth else ''}{n.title}"
                        f"{' ← current' if n.is_current else ''}",
                        n,
                    )
                    for n in nodes
                ],
                jump,
            )
            return True
        return False

    def _open_command_choices(
        self,
        choices: list[tuple[str, object]],
        on_select: Callable[[object], None],
    ) -> None:
        """打开可复用的 TUI 命令选择菜单。"""
        self._state.pending_command_choice = _CommandChoiceRequest(choices, on_select)
        self._command_choices.values = [(value, label) for label, value in choices]
        self._command_choices._selected_index = 0
        self._command_choices.current_value = choices[0][1]
        self._application.layout.focus(self._command_choices)
        self._refresh()

    def _open_command_text(self, prompt: str, on_submit: Callable[[str], None]) -> None:
        """打开单行文本表单。"""
        self._state.pending_command_text = _CommandTextRequest(prompt, on_submit)
        self._application.layout.focus(self._input)
        self._refresh()

    def _submit_command_text(self, text: str) -> None:
        request = self._state.pending_command_text
        if request is None:
            return
        self._state.pending_command_text = None
        request.on_submit(text)
        self._refresh()

    def _open_config_profile_editor(self) -> None:
        """在 TUI 内选择配置项并提交字段值。"""
        from ..setup_wizard import CONFIG_FILENAME, _load_existing_config

        config = _load_existing_config(self._project_root / CONFIG_FILENAME)
        profiles = config.get("provider", {}).get("model_profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            self._state.log.append(_LogEntry("system", "No profiles found."))
            self._refresh()
            return

        def choose_profile(profile: object) -> None:
            fields = [
                "transport",
                "chat_model",
                "base_url",
                "api_key",
                "thinking",
                "reasoning_effort",
                "clear_thinking",
                "tool_stream",
            ]

            def choose_field(field: object) -> None:
                profile_name = str(profile)
                field_name = str(field)
                self._open_command_text(
                    f"{profile_name}.{field_name}",
                    lambda value: (
                        self._run_command(
                            f"/config set {profile_name} {field_name} {value}"
                        )
                        if value
                        else None
                    ),
                )

            self._open_command_choices(
                [(field, field) for field in fields], choose_field
            )

        self._open_command_choices(
            [(str(name), name) for name in profiles], choose_profile
        )

    def _open_config_profile_delete(self) -> None:
        """在 TUI 内选择并删除配置 profile。"""
        from ..setup_wizard import CONFIG_FILENAME, _load_existing_config, _save_config

        config_path = self._project_root / CONFIG_FILENAME
        config = _load_existing_config(config_path)
        profiles = config.get("provider", {}).get("model_profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            self._state.log.append(_LogEntry("system", "No profiles found."))
            self._refresh()
            return

        def delete(profile: object) -> None:
            name = str(profile)
            profiles.pop(name, None)
            _save_config(config, config_path)
            self._state.log.append(_LogEntry("system", f"Deleted profile: {name}"))

        self._open_command_choices([(str(name), name) for name in profiles], delete)

    def _accept_command_choice(self) -> None:
        request = self._state.pending_command_choice
        if request is None:
            return
        selected = self._command_choices.current_value
        self._state.pending_command_choice = None
        request.on_select(selected)
        self._application.layout.focus(self._input)
        self._scrollback = 0
        self._refresh()

    def _finish_command(self, text: str, should_exit: bool) -> None:
        """同步命令执行后的 TUI 状态。"""
        self._state.mode = self._repl_state.mode
        if _is_session_history_command(text):
            self._restore_session_history()
        self._state.running = False
        self._refresh()
        if should_exit:
            print_saved_conversation(self._store)
            self._application.exit()
        self._submit_pending_inject()

    def _clear_session(self) -> None:
        """在 inline TUI 内创建空会话，不切换到终端清屏输出。"""
        self._store.clear()
        sync_agent_history(self._agent_app, self._store)
        self._state.restore_history([])
        self._state.log.append(_LogEntry("system", self._header_text()))
        self._scrollback = 0
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            agent.session_id = self._store.session_id
            from ..repl_commands import _compute_context_summary

            _compute_context_summary(agent, self._project_root, self._repl_state)
        self._refresh()

    def _run_shell_shortcut(self, text: str) -> None:
        """在后台执行 shell 快捷命令，保持 TUI 事件循环可响应。"""
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            token = getattr(agent, "cancellation_token", None)
            if token is not None:
                token.reset()
        self._state.running = True
        self._refresh()

        def run() -> None:
            try:
                output = run_shell_shortcut(text, self._agent_app)
                self._store.append("event", {"type": "shell_shortcut", "data": text})
                self._store.append("event", {"type": "tool_result", "data": output})
                self._state.log.append(_LogEntry("shell", output, markdown=False))
            except Exception as exc:
                detail = str(exc) or repr(exc)
                self._state.log.append(
                    _LogEntry("error", f"[error] {type(exc).__name__}: {detail}")
                )
            finally:
                self._state.running = False
                self._refresh()

        threading.Thread(target=run, daemon=True).start()

    def _restore_session_history(self) -> None:
        """恢复会话命令完成后，以 TUI 形式重新渲染当前分支。"""
        self._state.restore_history(self._store.build_branch())
        self._scrollback = 0
        agent = getattr(self._agent_app, "agent", None)
        if agent is not None:
            agent.session_id = self._store.session_id

    def _restore_startup_session(
        self,
        resume_latest: bool,
        auto_continue: bool,
        session_id: str | None,
    ) -> None:
        """在 TUI 创建前选择并恢复会话，避免以空白会话覆盖历史。"""
        selected = None
        if session_id is not None:
            selected = self._store.find_by_id(session_id)
            if selected is None:
                raise ValueError(f"Session not found: {session_id}")
            stored = (
                Path(selected.project_path).resolve() if selected.project_path else None
            )
            if stored is None or stored != self._project_root.resolve():
                raise ValueError(
                    f"Session belongs to another project: {selected.project_path}"
                )
        elif auto_continue:
            selected = self._store.find_latest_for_project(self._project_root)
        elif resume_latest:
            selected = select_session_interactively(
                self._store.list_infos(), "Select session to resume:"
            )
        if selected is None:
            return
        self._store.resume(selected.id)
        sync_agent_history(self._agent_app, self._store)
        self._state.restore_history(self._store.build_branch())

    # ── HITL ──

    def _approval_callback(self, tool: ToolSpec, action_input: ToolInput) -> HITLResult:
        from xcode.harness.observability import HITLResult

        request = _HitlRequest(
            tool_name=tool.name,
            preview=[
                _strip_rich_markup(line)
                for line in tool_preview_lines(tool, action_input)
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

    def _submit_pending_inject(self) -> None:
        """在命令设置注入内容后，以普通用户回合继续执行。"""
        text = self._repl_state.pending_inject
        self._repl_state.pending_inject = None
        if text:
            self._submit(text)

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
        thinking_parts: list[str] = []
        thinking_started_at: float | None = None

        def flush_thinking() -> None:
            nonlocal thinking_started_at
            if not thinking_parts:
                return
            duration_ms = int(
                (perf_counter() - thinking_started_at) * 1000
                if thinking_started_at is not None
                else 0
            )
            self._store.append(
                "event",
                {
                    "type": "thinking",
                    "data": {
                        "content": "".join(thinking_parts),
                        "duration_ms": duration_ms,
                    },
                },
            )
            thinking_parts.clear()
            thinking_started_at = None

        try:
            for event in self._agent_app.ask_stream(text, mode=self._repl_state.mode):
                if event.type == "reasoning_delta":
                    thinking_parts.append(event.data)
                    if thinking_started_at is None:
                        thinking_started_at = perf_counter()
                else:
                    flush_thinking()
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
            flush_thinking()
            self._save_partial_answer()
            detail = str(exc) or repr(exc)
            self._state.log.append(
                _LogEntry("error", f"[error] {type(exc).__name__}: {detail}")
            )
            self._state.running = False
            self._refresh()
            self._schedule_turn_commit()
            return

        flush_thinking()

        if answer:
            self._store.append("assistant", answer)
            self._store.update_summary()
        else:
            self._save_partial_answer()

        self._refresh()

        _exit_snapshot_ctx(snapshot, _snapshot_ctx, self._store.session_id, tool_names)
        self._schedule_turn_commit()

    def _schedule_turn_commit(self) -> None:
        """保留回合内容，允许完成后继续折叠和滚动查看。"""
        self._committing = False
        self._refresh()
        agent = getattr(self._agent_app, "agent", None)
        take_follow_up = getattr(agent, "take_follow_up", None)
        follow_up = take_follow_up() if callable(take_follow_up) else None
        if isinstance(follow_up, UserMessage):
            self._submit(str(follow_up.content))
            return
        self._submit_pending_inject()

    def _save_partial_answer(self) -> None:
        """将中断前已经流式显示的回答写入会话，供恢复和后续注入使用。"""
        partial = "".join(
            entry.text for entry in self._state.log if entry.role == "xcode"
        ).strip()
        if partial:
            self._store.append("assistant", partial)
            self._store.update_summary()

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
        input_height = self._input_height()
        input_visible = self._state.pending_command_choice is None and (
            self._state.pending_hitl is None or self._awaiting_denial_suggestion
        )
        command_height = (
            len(self._state.pending_command_choice.choices)
            if self._state.pending_command_choice is not None
            else 0
        )
        return max(
            1,
            self._application.output.get_size().rows
            - input_height
            - (2 if input_visible else 0)
            - 1
            - approval_height,
            -command_height,
        )

    def _input_height(self) -> int:
        return min(5, max(1, self._input.text.count("\n") + 1))

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

    def _status_text(self) -> str:
        left = f"mode: {self._repl_state.mode}"
        parts: list[str] = []
        if self._repl_state.context_usage:
            parts.append(f"context: {self._repl_state.context_usage}")
        if self._repl_state.context_cost:
            parts.append(f"cost: {self._repl_state.context_cost}")
        if not parts:
            return left
        right = "  ".join(parts)
        width = self._application.output.get_size().columns
        padding = max(2, width - len(left) - len(right))
        return f"{left}{' ' * padding}{right}"

    def _header_text(self) -> str:
        get_model_info = getattr(self._agent_app, "get_model_info", None)
        raw_info = get_model_info() if callable(get_model_info) else None
        info: dict[str, object] = raw_info if isinstance(raw_info, dict) else {}
        model = str(info.get("model") or self._repl_state.model_name or "unknown")
        effort = str(info.get("reasoning_effort") or "")
        model_display = f"{model} ({effort})" if effort else model
        return f"✦ Xcode\n· {model_display}\n: {self._project_root}"


# ── 模块级工具函数 ──


def _is_live_command(text: str) -> bool:
    """判断命令是否可以在 agent 回合执行期间提交。"""
    return text.split(maxsplit=1)[0] in {"/steer", "/queue"}


def _tui_history(project_root: Path) -> History | None:
    """为 TUI 输入栏创建与 REPL 相同的持久化历史记录。"""
    try:
        from prompt_toolkit.history import FileHistory

        history_dir = project_root / ".local"
        history_dir.mkdir(parents=True, exist_ok=True)
        return FileHistory(str(history_dir / "repl_history"))
    except OSError:
        return None


def _strip_rich_markup(text: str) -> str:
    """Rich 标记不能由 prompt_toolkit 解析，转换为保留内容的纯文本。"""
    return re.sub(r"\[/?[^\]]+]", "", text)


def _is_session_history_command(command: str) -> bool:
    """判断命令是否会切换或重写当前会话分支。"""
    name = command.split(maxsplit=1)[0]
    return name in {"/resume", "/continue", "/sessions", "/tree", "/rewind"}


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
