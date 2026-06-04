from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

from rich.console import Console
from rich.text import Text

from .commands import PromptLike, PromptText, ReplState
from .file_refs import expand_file_references
from .markdown import MarkdownRenderer, TerminalMarkdownRenderer
from .repl_commands import COMMAND_NAMES, handle_command
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
from .repl_sessions import print_saved_conversation, resume_interactively
from .repl_tools import (
    brief_input,
    event_to_dict,
    file_reference_event,
    final_stop_reason,
    print_tool_call_rich,
    print_tool_result_rich,
    summarize_intents,
    tool_intent,
)
from xcode.harness.observability import (
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.session import SessionStore


def run_repl(
    app: Any,
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
    session = prompt_session or create_prompt_session(root, registry, COMMAND_NAMES)
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
    app: Any,
    store: SessionStore,
    markdown_renderer: MarkdownRenderer,
    state: ReplState,
    session: PromptLike,
    agent_text: str,
) -> None:
    step_answers: list[str] = []
    current_step_thoughts: list[str] = []
    stopped_reason: str | None = None
    interrupted = False
    live_console = Console(file=sys.stdout)
    reasoning_started_at: float | None = None
    reasoning_text = ""
    reasoning_preview = LiveReasoningPreview(live_console)
    answer_stream = LiveMarkdownStream(live_console)
    streamed_text = False
    tool_group: dict[str, Any] | None = None
    tool_call_labels: dict[str, str] = {}

    def safe_write(text: str) -> None:
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

    def clear_line() -> None:
        safe_write("\r\033[K")

    def record_tool_call(event_data: Any) -> None:
        nonlocal tool_group
        label = brief_input(event_data.name, event_data.input)
        intent = tool_intent(event_data.name, event_data.input)
        tool_call_labels[event_data.id] = label
        if state.verbose:
            print_tool_call_rich(label, live_console)
            return
        if tool_group is None:
            tool_group = {
                "intents": [],
                "calls": 0,
                "ok": 0,
                "errors": [],
            }
        tool_group["calls"] += 1
        if intent not in tool_group["intents"]:
            tool_group["intents"].append(intent)

    def record_tool_result(event_data: Any) -> None:
        if state.verbose:
            print_tool_result_rich(event_data, state.verbose, live_console)
            return
        if tool_group is None:
            return
        if event_data.status == "ok":
            tool_group["ok"] += 1
            return
        label = tool_call_labels.get(event_data.tool_use_id, event_data.tool_use_id)
        tool_group["errors"].append((label, event_data))

    def flush_tool_group() -> None:
        nonlocal tool_group
        if tool_group is None:
            return
        calls = int(tool_group["calls"])
        errors = list(tool_group["errors"])
        intents = list(tool_group["intents"])
        title = summarize_intents(intents)
        status = "failed" if errors else "done"
        style = CLI_COLOR_ERROR if errors else CLI_COLOR_SUCCESS
        live_console.print(Text(f"  • Explore: {title}", style=CLI_COLOR_TOOL))
        live_console.print(Text(f"    {status}: {calls} tools", style=style))
        for label, result in errors:
            summary = single_line_preview(str(result.content), width=120)
            live_console.print(
                Text(f"    error: {label}: {summary}", style=CLI_COLOR_ERROR)
            )
        tool_group = None

    def render_reasoning_preview() -> None:
        lines = reasoning_preview_lines(reasoning_text)
        if lines:
            reasoning_preview.update(lines)

    def finish_reasoning_preview() -> None:
        nonlocal reasoning_started_at, reasoning_text
        if reasoning_started_at is None:
            return
        elapsed = time.perf_counter() - reasoning_started_at
        reasoning_preview.stop()
        if not should_print_reasoning_summary(reasoning_text, elapsed):
            reasoning_started_at = None
            reasoning_text = ""
            return
        preview = single_line_preview(reasoning_text)
        live_console.print(
            Text(
                f"  • thinked for {format_elapsed(elapsed)}",
                style=CLI_COLOR_THINKING,
            )
        )
        if preview:
            live_console.print(Text(f"    {preview}", style=CLI_COLOR_THINKING))
        reasoning_started_at = None
        reasoning_text = ""

    if state.pending_partial is not None:
        _, partial_text = state.pending_partial
        state.pending_partial = None
        if partial_text:
            agent_text = (
                f"[Context: assistant was interrupted mid-response]\n"
                f"{partial_text}\n"
                f"[end of partial response]\n"
                f"{agent_text}"
            )

    try:
        for event in app.ask_stream(agent_text, mode=state.mode):
            store.append("event", event_to_dict(event))
            if event.type == "reasoning_delta":
                flush_tool_group()
                if reasoning_started_at is None:
                    reasoning_started_at = time.perf_counter()
                reasoning_text += str(event.data)
                render_reasoning_preview()
                continue

            finish_reasoning_preview()

            if event.type == "text_delta":
                flush_tool_group()
                chunk = str(event.data)
                current_step_thoughts.append(chunk)
                streamed_text = True
                answer_stream.update("".join(current_step_thoughts))
            elif event.type == "assistant":
                if _assistant_has_tool_calls(event.data):
                    thoughts = "".join(current_step_thoughts).strip()
                    if thoughts:
                        answer_stream.stop()
                        if step_answers:
                            print()
                        step_answers.append(thoughts)
                    current_step_thoughts = []
                    streamed_text = False
            elif event.type == "tool_use":
                record_tool_call(event.data)
            elif event.type == "tool_result":
                record_tool_result(event.data)
            elif event.type == "final":
                flush_tool_group()
                final_answer = "".join(current_step_thoughts).strip()
                if not final_answer and getattr(event.data, "answer", None):
                    final_answer = str(event.data.answer).strip()
                if final_answer:
                    step_answers.append(final_answer)
                    if streamed_text:
                        answer_stream.stop()
                    elif len(step_answers) > 1:
                        print()
                        markdown_renderer.render(final_answer)
                    else:
                        markdown_renderer.render(final_answer)
                    streamed_text = False
                stopped_reason = final_stop_reason(event.data)
    except KeyboardInterrupt:
        interrupted = True
        token = getattr(getattr(app, "agent", None), "cancellation_token", None)
        if token is not None:
            token.cancel("interrupted by user")
        tool_group = None
        clear_line()
        store.append("event", {"type": "interrupted", "data": "interrupted by user"})
        has_partial = bool(reasoning_text or current_step_thoughts or step_answers)
        if has_partial:
            state.pending_partial = (
                reasoning_text,
                "".join(current_step_thoughts)
                or ("".join(step_answers) if step_answers else ""),
            )
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
        finish_reasoning_preview()
        answer_stream.stop()
        if not interrupted:
            flush_tool_group()

    if stopped_reason:
        print(stopped_reason)
    if step_answers:
        answer = "\n\n".join(step_answers)
        store.append("assistant", answer)
        store.update_summary()


def _assistant_has_tool_calls(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use" for block in data
    )
