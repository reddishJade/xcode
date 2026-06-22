from __future__ import annotations

import queue
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import cast

from rich.console import Console
from rich.text import Text

from .app_contract import ReplApp
from .commands import PromptLike, PromptText, ReplState
from .file_refs import expand_file_references
from .markdown import MarkdownRenderer, TerminalMarkdownRenderer
from .repl_commands import COMMAND_NAMES, COMMAND_REGISTRY_EXPORT, handle_command
from .repl_hitl import ReplHITLHandler
from .repl_rendering import (
    LiveMarkdownStream,
    CLI_COLOR_INFO,
    # clear_terminal_display,
    create_prompt_session,
    input_prompt,
    print_startup_banner,
)
from .repl_turn_handler import ReasoningHandler, ToolCallHandler
from .reasoning_effort import reasoning_effort_levels_for_transport
from xcode.ai.registry import get_models, get_providers
from .repl_sessions import (
    print_saved_conversation,
    resume_interactively,
    sync_agent_history,
)
from .repl_skills import (
    activate_skill,
    available_skill_names,
    parse_skill_invocation,
)
from .repl_tools import (
    event_to_dict,
    file_reference_event,
    final_stop_reason,
    run_shell_shortcut,
)
from xcode.harness.agent_runtime.events import (
    AssistantEventBlock,
    AssistantStructuredEvent,
    AssistantToolUseBlock,
    FinalStructuredEvent,
    ReasoningDeltaStructuredEvent,
    StructuredAgentEvent,
    TextDeltaStructuredEvent,
    ToolResultStructuredEvent,
    TodoUpdateStructuredEvent,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.harness.agent_runtime.result import StructuredAgentResult
from xcode.harness.observability import FileGrantStore
from xcode.harness.observability.permission_model import SessionGrantStoreManager
from xcode.harness.session import SessionMetadataView, SessionStore
from xcode.harness.session_todo import TodoItem
from xcode.harness.snapshot import (
    SnapshotResult,
    SnapshotStore,
    SnapshotUnsupportedError,
)
from xcode.agent.messages import UserMessage


def current_effort_options(app: object) -> tuple[str, ...]:
    """返回当前 active provider 支持的 reasoning effort 选项。"""
    agent = getattr(app, "agent", None)
    provider = getattr(agent, "provider", None) if agent else None
    provider = getattr(provider, "active_provider", provider)
    transport = getattr(provider, "transport", "") if provider else ""
    return reasoning_effort_levels_for_transport(transport)


def current_model_options(app: object) -> tuple[str, ...]:
    """返回所有注册的模型 ID 列表（含当前模型，即便非预设）。"""
    all_models: list[str] = []
    for provider_name in get_providers():
        all_models.extend(m.id for m in get_models(provider_name))
    agent = getattr(app, "agent", None)
    provider = getattr(agent, "provider", None) if agent else None
    provider = getattr(provider, "active_provider", provider)
    current_model = getattr(provider, "model", "") if provider else ""
    if current_model and current_model not in all_models:
        all_models.append(current_model)
    return tuple(all_models)


def run_repl(
    app: ReplApp,
    sessions_dir: Path,
    prompt_session: PromptLike | None = None,
    resume_latest: bool = False,
    auto_continue: bool = False,
    session_id: str | None = None,
    renderer: MarkdownRenderer | None = None,
    project_root: Path | None = None,
) -> int:
    root = (project_root or sessions_dir).resolve()
    store = SessionStore(sessions_dir, project_root=root)
    markdown_renderer = renderer or TerminalMarkdownRenderer()
    registry = tuple(getattr(app, "registry", ()) or ())

    state = ReplState()

    use_governance = bool(getattr(getattr(app, "agent", None), "use_governance", False))
    session = prompt_session or create_prompt_session(
        root,
        registry,
        COMMAND_NAMES,
        COMMAND_REGISTRY_EXPORT,
        lambda: current_effort_options(app),
        lambda: current_model_options(app),
        lambda: available_skill_names(app),
        state=state,
        use_registered_tool_governance=use_governance,
    )
    grant_store_manager = SessionGrantStoreManager()
    permanent_grant_store = FileGrantStore.for_project_root(root)
    hitl_handler = ReplHITLHandler(session)
    snapshot_store: SnapshotStore | None = None
    try:
        snapshot_store = SnapshotStore(root)
    except SnapshotUnsupportedError:
        pass
    agent = getattr(app, "agent", None)
    if agent is not None:
        agent.approval_callback = hitl_handler
        agent.session_id = store.session_id
        if hasattr(agent, "set_session_grant_store_provider"):
            agent.set_session_grant_store_provider(
                lambda: grant_store_manager.get_for_session(store.session_id)
            )
        if hasattr(agent, "set_permanent_grant_store"):
            agent.set_permanent_grant_store(permanent_grant_store)
    # clear_terminal_display()  # 在打印启动横幅前清理屏幕，保持界面整洁
    print_startup_banner(app, root)

    # 初始化底栏上下文摘要
    if agent is not None:
        from .repl_commands import _compute_context_summary

        _compute_context_summary(agent, root, state)

    # session selection phase — exactly one path executes
    selected_view: SessionMetadataView | None = None

    if session_id is not None:
        view = store.find_by_id(session_id)
        if view is None:
            print(f"Session not found: {session_id}", file=sys.stderr)
            return 1
        stored = Path(view.project_path).resolve() if view.project_path else None
        if stored is None or str(stored) != str(root.resolve()):
            print(
                f"Session belongs to another project: {view.project_path}",
                file=sys.stderr,
            )
            return 1
        store.resume(session_id)
        selected_view = view

    elif auto_continue:
        view = store.find_latest_for_project(root)
        if view is not None:
            store.resume(view.id)
            selected_view = view
        else:
            print("No prior session found for this project. Starting a new session.")

    elif resume_latest:
        resume_interactively(store, session)
        sync_agent_history(app, store)

    if selected_view is not None:
        from .repl_sessions import resumed_message, print_loaded_history

        print(resumed_message(selected_view))
        print_loaded_history(store)
        sync_agent_history(app, store)

    while True:
        if state.pending_inject is not None:
            text = state.pending_inject
            state.pending_inject = None
        else:
            try:
                prompt_text: PromptText = (
                    ""
                    if state.exit_pending and time.time() - state.exit_pending < 1.5
                    else input_prompt()
                )
                text = session.prompt(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                now = time.time()
                if state.exit_pending and now - state.exit_pending < 1.5:
                    print()
                    try:
                        print_saved_conversation(store)
                    except KeyboardInterrupt:
                        pass
                    return 0
                state.exit_pending = now
                print()
                sys.stdout.write("\033[90m(press Ctrl+C again to exit)\033[0m\n")
                sys.stdout.flush()
                continue
        if not text:
            continue
        state.exit_pending = 0.0
        skill_invocation = parse_skill_invocation(text)
        if skill_invocation is not None:
            skill_name, remaining_text = skill_invocation
            activation = activate_skill(app, store, skill_name)
            print(activation.message)
            if activation.status not in {"activated", "already_active"}:
                continue
            if not remaining_text:
                continue
            text = remaining_text
        if text.startswith("/"):
            current_session_store = grant_store_manager.get_for_session(
                store.session_id
            )
            if handle_command(
                text,
                store,
                app,
                markdown_renderer,
                state,
                session,
                current_session_store,
                permanent_grant_store,
                static_policy=getattr(agent, "permission_policy", None),
                restricted_dirs=getattr(agent, "restricted_dirs", ()),
                snapshot_store=snapshot_store,
            ):
                print_saved_conversation(store)
                return 0
            continue
        if text.startswith("!"):
            output = run_shell_shortcut(text, app)
            store.append("event", {"type": "shell_shortcut", "data": text})
            store.append("event", {"type": "tool_result", "data": output})
            _print_raw_output(output)
            continue

        store.append("user", text)

        # 用户轮次边界快照：拍摄 pre 快照，执行 agent 轮次，finally 中拍 post 快照
        snapshot_turn_id: str | None = None
        pre_result: SnapshotResult | None = None
        if snapshot_store is not None:
            try:
                snapshot_turn_id = snapshot_store.next_turn_id(store.session_id)
                svc = snapshot_store.service(store.session_id)
                pre_result = svc.track()
            except KeyboardInterrupt:
                print("[interrupted] current run cancelled; session is still active.")
                continue

        expanded_text, references = expand_file_references(text, root)
        if references:
            store.append("event", file_reference_event(references))
        agent_text = expanded_text
        if state.approved_plan is not None:
            agent_text = (
                f"<approved-plan>\n{state.approved_plan}\n</approved-plan>\n"
                f"{expanded_text}"
            )
            state.approved_plan = None
        ctx = _AgentTurnContext(
            app=app,
            store=store,
            renderer=markdown_renderer,
            state=state,
            session=session,
            text=agent_text,
        )
        tool_names: list[str] = []
        try:
            tool_names = _run_agent_turn(ctx)
        finally:
            if snapshot_store is not None and pre_result is not None:
                try:
                    svc = snapshot_store.service(store.session_id)
                    post_result = svc.track()
                    changes = svc.diff(
                        pre_result.snapshot_id,
                        post_result.snapshot_id,
                    )
                    all_skipped = list(pre_result.skipped_files)
                    all_skipped.extend(post_result.skipped_files)
                    snapshot_store.record_turn(
                        session_id=store.session_id,
                        turn_id=snapshot_turn_id or "000",
                        pre_snapshot_id=pre_result.snapshot_id,
                        post_snapshot_id=post_result.snapshot_id,
                        changed_files=changes,
                        skipped_files=all_skipped,
                        tool_names=tool_names,
                    )
                except KeyboardInterrupt:
                    print("[interrupted] snapshot cancelled; session is still active.")


@dataclass
class _AgentTurnContext:
    app: ReplApp
    store: SessionStore
    renderer: MarkdownRenderer
    state: ReplState
    session: PromptLike
    text: str


def _run_agent_turn(ctx: _AgentTurnContext) -> list[str]:
    turn = _ReplTurnRenderer(ctx.renderer, ctx.state)
    queued_input = _start_queue_input_reader(ctx.state, ctx.session)

    text = ctx.text
    if ctx.state.pending_partial is not None:
        _, partial_text = ctx.state.pending_partial
        ctx.state.pending_partial = None
        if partial_text:
            agent = getattr(ctx.app, "agent", None)
            steer = getattr(agent, "steer", None)
            if callable(steer):
                from xcode.agent.messages import AssistantMessage
                from xcode.agent.types import TextContent

                steer(
                    AssistantMessage(
                        content=[
                            TextContent(
                                text="[assistant was interrupted mid-response]\n"
                                f"{partial_text}"
                            )
                        ]
                    )
                )
            else:
                text = (
                    f"[Context: assistant was interrupted mid-response]\n"
                    f"{partial_text}\n"
                    f"[end of partial response]\n"
                    f"{text}"
                )

    try:
        ask_stream = getattr(ctx.app, "ask_stream", None)
        if not callable(ask_stream):
            raise TypeError("app does not support ask_stream")
        typed_ask_stream = cast(
            Callable[..., Iterator[StructuredAgentEvent]], ask_stream
        )
        for event in typed_ask_stream(text, mode=ctx.state.mode):
            ctx.store.append("event", event_to_dict(event))
            turn.handle_event(event)
    except KeyboardInterrupt:
        turn.interrupted = True
        token = getattr(getattr(ctx.app, "agent", None), "cancellation_token", None)
        if token is not None:
            token.cancel("interrupted by user")
        turn.clear_line()
        ctx.store.append(
            "event", {"type": "interrupted", "data": "interrupted by user"}
        )
        reasoning_text, partial_answer = turn.partial_state()
        has_partial = bool(reasoning_text or partial_answer)
        if has_partial:
            ctx.state.pending_partial = (reasoning_text, partial_answer)
            print("[interrupted] type your message below to inject into context")
            try:
                inject_text = ctx.session.prompt(
                    [("class:prompt-marker", "interrupt> "), ("", "")]
                ).strip()
            except (EOFError, KeyboardInterrupt):
                inject_text = ""
            if inject_text:
                ctx.state.pending_inject = inject_text
                ctx.store.append("user", inject_text)
            else:
                ctx.state.pending_partial = None
                print("[interrupt cancelled]")
        else:
            print("[interrupted] current run cancelled; session is still active.")
        return turn.tool_names_in_turn
    finally:
        queued_lines = queued_input.drain() if queued_input is not None else []
        if queued_lines:
            agent = getattr(ctx.app, "agent", None)
            for queued_line in queued_lines:
                ctx.store.append("user", queued_line)
                if agent is not None and hasattr(agent, "follow_up"):
                    agent.follow_up(UserMessage(content=queued_line))
            print(f"[queued] {len(queued_lines)} follow-up message(s)")
        turn.close()

    if turn.stopped_reason:
        print(turn.stopped_reason)
    if turn.step_answers:
        answer = "\n\n".join(turn.step_answers)
        ctx.store.append("assistant", answer)
        ctx.store.update_summary()
    return turn.tool_names_in_turn


class _QueuedInput:
    def __init__(
        self,
        thread: threading.Thread,
        lines: queue.SimpleQueue[str],
    ) -> None:
        self.thread = thread
        self.lines = lines

    def drain(self) -> list[str]:
        self.thread.join()
        drained: list[str] = []
        while True:
            try:
                drained.append(self.lines.get_nowait())
            except queue.Empty:
                return drained


def _start_queue_input_reader(
    state: ReplState,
    session: PromptLike,
) -> _QueuedInput | None:
    if not state.queue_mode:
        return None
    lines: queue.SimpleQueue[str] = queue.SimpleQueue()

    def read_lines() -> None:
        while True:
            try:
                text = session.prompt(
                    [("class:prompt-marker", "queue> "), ("", "")]
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not text:
                return
            lines.put(text)

    thread = threading.Thread(target=read_lines, daemon=True)
    thread.start()
    return _QueuedInput(thread, lines)


class _ReplTurnRenderer:
    """协调 agent 回合的事件流式渲染，委托给专门的处理器处理推理、工具和答案输出。"""

    def __init__(self, markdown_renderer: MarkdownRenderer, state: ReplState) -> None:
        self.markdown_renderer = markdown_renderer
        self.state = state
        self.step_answers: list[str] = []
        self.current_step_thoughts: list[str] = []
        self.stopped_reason: str | None = None
        self.interrupted = False
        self.live_console = Console(file=sys.stdout)
        self.answer_stream = LiveMarkdownStream(self.live_console)
        self.streamed_text = False
        self.tool_handler = ToolCallHandler(state, self.live_console)
        self.reasoning_handler = ReasoningHandler(
            self.live_console, self.state.verbosity
        )
        self.tool_names_in_turn: list[str] = []

    def handle_event(self, event: StructuredAgentEvent) -> None:
        if isinstance(event, ReasoningDeltaStructuredEvent):
            self.tool_handler.flush_group()
            self.reasoning_handler.handle_delta(event.data)
            return

        self.reasoning_handler.finish()
        if isinstance(event, TextDeltaStructuredEvent):
            self._handle_text_delta(event.data)
        elif isinstance(event, AssistantStructuredEvent):
            self._handle_assistant_event(event.data)
        elif isinstance(event, ToolUseStructuredEvent):
            self.tool_handler.record_tool_call(event.data)
            self.tool_names_in_turn.append(event.data.name)
        elif isinstance(event, ToolUpdateStructuredEvent):
            self.tool_handler.handle_tool_update(event.data)
        elif isinstance(event, ToolResultStructuredEvent):
            self.tool_handler.record_tool_result(event.data)
            self.tool_handler.clear_progress()
        elif isinstance(event, TodoUpdateStructuredEvent):
            self._handle_todo_update(event.data)
        elif isinstance(event, FinalStructuredEvent):
            self._handle_final_event(event.data)

    def partial_state(self) -> tuple[str, str]:
        partial_answer = "".join(self.current_step_thoughts) or (
            "".join(self.step_answers) if self.step_answers else ""
        )
        return self.reasoning_handler.text, partial_answer

    def close(self) -> None:
        self.reasoning_handler.finish()
        self.answer_stream.stop()
        if not self.interrupted:
            self.tool_handler.flush_group()
        else:
            self.tool_handler.discard_group()

    def clear_line(self) -> None:
        self.tool_handler.clear_line()

    def _handle_text_delta(self, event_data: str) -> None:
        self.tool_handler.flush_group()
        self.current_step_thoughts.append(event_data)
        self.streamed_text = True
        self.answer_stream.update("".join(self.current_step_thoughts))

    def _handle_assistant_event(
        self, event_data: tuple[AssistantEventBlock, ...]
    ) -> None:
        if not _assistant_has_tool_calls(event_data):
            return
        thoughts = "".join(self.current_step_thoughts).strip()
        if thoughts:
            self.answer_stream.stop()
            if self.step_answers:
                print()
            self.step_answers.append(thoughts)
        self.current_step_thoughts = []
        self.streamed_text = False

    def _handle_final_event(self, event_data: StructuredAgentResult) -> None:
        self.tool_handler.flush_group()
        final_answer = "".join(self.current_step_thoughts).strip()
        if not final_answer and event_data.answer:
            final_answer = event_data.answer.strip()
        if final_answer:
            self.step_answers.append(final_answer)
            if self.streamed_text:
                self.answer_stream.stop()
            elif len(self.step_answers) > 1:
                print()
                self.markdown_renderer.render(final_answer)
            else:
                self.markdown_renderer.render(final_answer)
            self.streamed_text = False
        self.stopped_reason = final_stop_reason(event_data)

    def _handle_todo_update(self, items: tuple[TodoItem, ...]) -> None:
        """渲染当前会话待办清单。"""
        self.tool_handler.flush_group()
        self.live_console.print(
            Text(f"  Todo ({len(items)} items)", style=CLI_COLOR_INFO)
        )
        markers = {
            "pending": " ",
            "in_progress": "/",
            "completed": "x",
        }
        for item in items:
            self.live_console.print(
                Text(
                    f"    [{markers[item.status]}] {item.content}", style=CLI_COLOR_INFO
                )
            )


def _assistant_has_tool_calls(
    data: tuple[AssistantEventBlock, ...],
) -> bool:
    return any(isinstance(block, AssistantToolUseBlock) for block in data)


def _print_raw_output(text: str) -> None:
    if not text:
        return
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
