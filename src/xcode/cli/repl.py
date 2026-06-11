from __future__ import annotations

import queue
from pathlib import Path
import sys
import threading
import time
from typing import Any

from rich.console import Console
from rich.text import Text

from .commands import PromptLike, PromptText, ReplState
from .file_refs import expand_file_references
from .markdown import MarkdownRenderer, TerminalMarkdownRenderer
from .repl_commands import COMMAND_NAMES, COMMAND_REGISTRY_EXPORT, handle_command
from .repl_hitl import ReplHITLHandler
from .repl_rendering import (
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_THINKING,
    CLI_COLOR_TOOL,
    LiveMarkdownStream,
    LiveReasoningPreview,
    create_prompt_session,
    format_elapsed,
    input_prompt,
    print_startup_banner,
    reasoning_preview_lines,
    should_print_reasoning_summary,
    single_line_preview,
)
from .reasoning_effort import reasoning_effort_levels_for_transport
from xcode.ai.registry import get_models, get_providers
from .repl_sessions import (
    print_saved_conversation,
    resume_interactively,
    sync_agent_history,
)
from .repl_tools import (
    brief_input,
    event_to_dict,
    file_reference_event,
    final_stop_reason,
    print_tool_call_rich,
    print_tool_result_rich,
    run_shell_shortcut,
    summarize_intents,
    tool_intent,
)
from xcode.harness.agent_runtime.event_translation import (
    AssistantEventBlock,
    AssistantStructuredEvent,
    AssistantToolUseBlock,
    FinalStructuredEvent,
    ReasoningDeltaStructuredEvent,
    StructuredAgentEvent,
    TextDeltaStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    ToolUpdateData,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.ai.events import ToolCall
from xcode.harness.agent_runtime.result import StructuredAgentResult
from xcode.harness.observability import (
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.app import XcodeApp
from xcode.harness.session import SessionStore
from xcode.agent.messages import UserMessage


def current_effort_options(app: XcodeApp) -> tuple[str, ...]:
    """返回当前 active provider 支持的 reasoning effort 选项。"""
    agent = app.agent
    provider = agent.provider if agent else None
    provider = getattr(provider, "active_provider", provider)
    transport = getattr(provider, "transport", "") if provider else ""
    return reasoning_effort_levels_for_transport(transport)


def current_model_options(app: XcodeApp) -> tuple[str, ...]:
    """返回所有注册的模型 ID 列表（含当前模型，即便非预设）。"""
    all_models: list[str] = []
    for provider_name in get_providers():
        all_models.extend(m.id for m in get_models(provider_name))
    agent = app.agent
    provider = agent.provider if agent else None
    provider = getattr(provider, "active_provider", provider)
    current_model = getattr(provider, "model", "") if provider else ""
    if current_model and current_model not in all_models:
        all_models.append(current_model)
    return tuple(all_models)


def run_repl(
    app: XcodeApp,
    sessions_dir: Path,
    prompt_session: PromptLike | None = None,
    resume_latest: bool = False,
    renderer: MarkdownRenderer | None = None,
    project_root: Path | None = None,
) -> int:
    root = (project_root or sessions_dir).resolve()
    store = SessionStore(sessions_dir, project_root=root)
    markdown_renderer = renderer or TerminalMarkdownRenderer()
    registry = tuple(getattr(app, "registry", ()) or ())

    session = prompt_session or create_prompt_session(
        root,
        registry,
        COMMAND_NAMES,
        COMMAND_REGISTRY_EXPORT,
        lambda: current_effort_options(app),
        lambda: current_model_options(app),
    )
    state = ReplState()
    session_policy = SessionPermissionPolicy()
    persistent_store = PersistentPermissionStore(root / ".local" / "hitl_policy.json")
    hitl_handler = ReplHITLHandler(session_policy, persistent_store, session)
    agent = getattr(app, "agent", None)
    if agent is not None:
        agent.approval_callback = hitl_handler
    print_startup_banner(app, root)
    if resume_latest:
        resume_interactively(store, session)
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
                    print_saved_conversation(store)
                    return 0
                state.exit_pending = now
                print()
                sys.stdout.write("\033[90m(press Ctrl+C again to exit)\033[0m\n")
                sys.stdout.flush()
                continue
        if not text:
            continue
        state.exit_pending = 0.0
        if text.startswith("/"):
            if handle_command(
                text,
                store,
                app,
                markdown_renderer,
                state,
                session,
                session_policy,
                persistent_store,
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
        _run_agent_turn(app, store, markdown_renderer, state, session, agent_text)


def _run_agent_turn(
    app: XcodeApp,
    store: SessionStore,
    markdown_renderer: MarkdownRenderer,
    state: ReplState,
    session: PromptLike,
    agent_text: str,
) -> None:
    turn = _ReplTurnRenderer(markdown_renderer, state)
    queued_input = _start_queue_input_reader(state, session)

    if state.pending_partial is not None:
        _, partial_text = state.pending_partial
        state.pending_partial = None
        if partial_text:
            agent = getattr(app, "agent", None)
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
                agent_text = (
                    f"[Context: assistant was interrupted mid-response]\n"
                    f"{partial_text}\n"
                    f"[end of partial response]\n"
                    f"{agent_text}"
                )

    try:
        for event in app.ask_stream(agent_text, mode=state.mode):
            store.append("event", event_to_dict(event))
            turn.handle_event(event)
    except KeyboardInterrupt:
        turn.interrupted = True
        token = getattr(getattr(app, "agent", None), "cancellation_token", None)
        if token is not None:
            token.cancel("interrupted by user")
        turn.clear_line()
        store.append("event", {"type": "interrupted", "data": "interrupted by user"})
        reasoning_text, partial_answer = turn.partial_state()
        has_partial = bool(reasoning_text or partial_answer)
        if has_partial:
            state.pending_partial = (reasoning_text, partial_answer)
            print("[interrupted] type your message below to inject into context")
            try:
                inject_text = session.prompt(
                    [("class:prompt-marker", "interrupt> "), ("", "")]
                ).strip()
            except (EOFError, KeyboardInterrupt):
                inject_text = ""
            if inject_text:
                state.pending_inject = inject_text
                store.append("user", inject_text)
            else:
                state.pending_partial = None
                print("[interrupt cancelled]")
        else:
            print("[interrupted] current run cancelled; session is still active.")
        return
    finally:
        queued_lines = queued_input.drain() if queued_input is not None else []
        if queued_lines:
            agent = getattr(app, "agent", None)
            for queued_line in queued_lines:
                store.append("user", queued_line)
                if agent is not None and hasattr(agent, "follow_up"):
                    agent.follow_up(UserMessage(content=queued_line))
            print(f"[queued] {len(queued_lines)} follow-up message(s)")
        turn.close()

    if turn.stopped_reason:
        print(turn.stopped_reason)
    if turn.step_answers:
        answer = "\n\n".join(turn.step_answers)
        store.append("assistant", answer)
        store.update_summary()


class _QueuedInput:
    def __init__(
        self,
        thread: threading.Thread,
        lines: queue.SimpleQueue[str],
    ) -> None:
        self.thread = thread
        self.lines = lines

    def drain(self) -> list[str]:
        self.thread.join(timeout=0.1)
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
    def __init__(self, markdown_renderer: MarkdownRenderer, state: ReplState) -> None:
        self.markdown_renderer = markdown_renderer
        self.state = state
        self.step_answers: list[str] = []
        self.current_step_thoughts: list[str] = []
        self.stopped_reason: str | None = None
        self.interrupted = False
        self.live_console = Console(file=sys.stdout)
        self.reasoning_started_at: float | None = None
        self.reasoning_text = ""
        self.reasoning_preview = LiveReasoningPreview(self.live_console)
        self.answer_stream = LiveMarkdownStream(self.live_console)
        self.streamed_text = False
        self.tool_group: dict[str, Any] | None = None
        self.tool_call_labels: dict[str, str] = {}
        self._progress_tool_id: str | None = None

    def handle_event(self, event: StructuredAgentEvent) -> None:
        if isinstance(event, ReasoningDeltaStructuredEvent):
            self._handle_reasoning_delta(event.data)
            return

        self._finish_reasoning_preview()
        if isinstance(event, TextDeltaStructuredEvent):
            self._handle_text_delta(event.data)
        elif isinstance(event, AssistantStructuredEvent):
            self._handle_assistant_event(event.data)
        elif isinstance(event, ToolUseStructuredEvent):
            self._record_tool_call(event.data)
        elif isinstance(event, ToolUpdateStructuredEvent):
            self._handle_tool_update(event.data)
        elif isinstance(event, ToolResultStructuredEvent):
            self._record_tool_result(event.data)
            self._clear_progress()
        elif isinstance(event, FinalStructuredEvent):
            self._handle_final_event(event.data)

    def partial_state(self) -> tuple[str, str]:
        partial_answer = "".join(self.current_step_thoughts) or (
            "".join(self.step_answers) if self.step_answers else ""
        )
        return self.reasoning_text, partial_answer

    def close(self) -> None:
        self._finish_reasoning_preview()
        self.answer_stream.stop()
        if not self.interrupted:
            self._flush_tool_group()
        else:
            self.tool_group = None

    def clear_line(self) -> None:
        self._safe_write("\r\033[K")

    def _handle_reasoning_delta(self, event_data: str) -> None:
        self._flush_tool_group()
        if self.reasoning_started_at is None:
            self.reasoning_started_at = time.perf_counter()
        self.reasoning_text += event_data
        lines = reasoning_preview_lines(self.reasoning_text)
        if lines:
            self.reasoning_preview.update(lines)

    def _handle_text_delta(self, event_data: str) -> None:
        self._flush_tool_group()
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
        self._flush_tool_group()
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

    def _record_tool_call(self, event_data: ToolCall) -> None:
        label = brief_input(event_data.name, event_data.input)
        intent = tool_intent(event_data.name, event_data.input)
        self.tool_call_labels[event_data.id] = label
        if self.state.verbose:
            print_tool_call_rich(label, self.live_console)
            return
        if self.tool_group is None:
            self.tool_group = {
                "intents": [],
                "calls": 0,
                "ok": 0,
                "errors": [],
            }
        self.tool_group["calls"] += 1
        if intent not in self.tool_group["intents"]:
            self.tool_group["intents"].append(intent)

    def _record_tool_result(self, event_data: ToolResultBlock) -> None:
        if self.state.verbose:
            print_tool_result_rich(event_data, self.state.verbose, self.live_console)
            return
        if self.tool_group is None:
            return
        if event_data.status == "ok":
            self.tool_group["ok"] += 1
            return
        label = self.tool_call_labels.get(
            event_data.tool_use_id, event_data.tool_use_id
        )
        self.tool_group["errors"].append((label, event_data))

    def _clear_progress(self) -> None:
        if self._progress_tool_id is not None:
            self.clear_line()
            self._progress_tool_id = None

    def _handle_tool_update(self, event_data: ToolUpdateData) -> None:
        tool_id = event_data.tool_call_id
        partial = event_data.partial_result
        if not tool_id or not partial:
            return
        if self._progress_tool_id != tool_id:
            self._clear_progress()
            self._progress_tool_id = tool_id
        lines = [line for line in partial.splitlines() if line.strip()]
        last_line = lines[-1] if lines else ""
        if len(last_line) > 100:
            last_line = last_line[:97] + "..."
        if last_line:
            self._safe_write(f"\r\033[K\x1b[90m  {last_line}\x1b[0m")

    def _flush_tool_group(self) -> None:
        if self.tool_group is None:
            return
        self._clear_progress()
        calls = int(self.tool_group["calls"])
        errors = list(self.tool_group["errors"])
        intents = list(self.tool_group["intents"])
        title = summarize_intents(intents)
        status = "failed" if errors else "done"
        style = CLI_COLOR_ERROR if errors else CLI_COLOR_SUCCESS
        self.live_console.print(Text(f"  • Explore: {title}", style=CLI_COLOR_TOOL))
        self.live_console.print(Text(f"    {status}: {calls} tools", style=style))
        for label, result in errors:
            summary = single_line_preview(str(result.content), width=120)
            self.live_console.print(
                Text(f"    error: {label}: {summary}", style=CLI_COLOR_ERROR)
            )
        self.tool_group = None

    def _finish_reasoning_preview(self) -> None:
        if self.reasoning_started_at is None:
            return
        elapsed = time.perf_counter() - self.reasoning_started_at
        self.reasoning_preview.stop()
        if not should_print_reasoning_summary(self.reasoning_text, elapsed):
            self.reasoning_started_at = None
            self.reasoning_text = ""
            return
        preview = single_line_preview(self.reasoning_text)
        self.live_console.print(
            Text(
                f"  • thinked for {format_elapsed(elapsed)}",
                style=CLI_COLOR_THINKING,
            )
        )
        if preview:
            self.live_console.print(Text(f"    {preview}", style=CLI_COLOR_THINKING))
        self.reasoning_started_at = None
        self.reasoning_text = ""

    def _safe_write(self, text: str) -> None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except UnicodeEncodeError:
            safe_text = (
                text.replace("•", "*")
                .replace("×", "x")
                .replace("✘", "x")
                .replace("⊘", "o")
            )
            encoding = sys.stdout.encoding or "utf-8"
            sys.stdout.write(
                safe_text.encode(encoding, errors="replace").decode(encoding)
            )
            sys.stdout.flush()


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
