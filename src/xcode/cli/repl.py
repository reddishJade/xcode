from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import shutil
import sys
import textwrap
import time
from typing import Any, Protocol

from .completion import ReplCompleter
from .file_refs import FileReference, expand_file_references
from .markdown import MarkdownRenderer, TerminalMarkdownRenderer
from .session import FORK_TYPES, SessionMetadataView, SessionStore
from xcode.harness.observability import (
    HITLResult,
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.skills import ToolSpec, run_tool_result

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

_console = Console(file=sys.stdout)
_MIN_REASONING_PREVIEW_SECONDS = 0.15
_THINKING_STYLE = "grey50"


class PromptLike(Protocol):
    def prompt(self, prompt_text: str) -> str: ...


def _reasoning_preview_lines(text: str, width: int | None = None) -> list[str]:
    width = width or max(20, shutil.get_terminal_size((100, 20)).columns - 4)
    lines: list[str] = []
    for line in text.splitlines() or [text]:
        wrapped = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines[-3:]


def _format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _single_line_preview(text: str, width: int | None = None) -> str:
    width = width or max(20, shutil.get_terminal_size((100, 20)).columns - 6)
    preview = " ".join(text.split())
    if len(preview) <= width:
        return preview
    return f"{preview[: max(0, width - 1)]}…"


class _LiveMarkdownStream:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.live: Live | None = None

    def update(self, text: str) -> None:
        if self.live is None:
            self.live = Live(
                Markdown(text),
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self.live.start(refresh=True)
            return
        self.live.update(Markdown(text), refresh=True)

    def stop(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None


class _LiveReasoningPreview:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.live: Live | None = None

    def update(self, lines: list[str]) -> None:
        text = Text("\n".join(lines), style=_THINKING_STYLE)
        if self.live is None:
            self.live = Live(
                text,
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self.live.start(refresh=True)
            return
        self.live.update(text, refresh=True)

    def stop(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None


HELP_TEXT = """Commands:
  /help      Show this help.
  /clear     Start a new session transcript.
  /fork [explore|verify|isolate]
             Fork current session into an independent branch.
  /rewind N  Remove the last N user turns from the transcript.
  /resume    Choose a recent conversation to resume.
  /resume last
             Resume the latest conversation.
  /sessions  List recent conversations.
  /model     Show current model info.
  /model <name>
             Switch model (e.g. /model deepseek-v4-pro).
  /model <name> --thinking <level>
             Switch model with thinking mode (off/minimal/low/medium/high/xhigh).
  /effort    Show current reasoning effort.
  /effort <level>
             Set reasoning effort (off/minimal/low/medium/high/max).
  /plan      Enter Plan Mode: read-only inspection tools, no edits or shell.
  /review    Enter Review Mode: read-only review, guarded validation.
  /act       Enter Act Mode and allow normal tool use within policy.
  /verbose on / off
             Show or hide tool call ids and result details.
  /compact   Manually request context compaction and shrink the session log.
  /permissions
             List session-level and persistent permission rules.
  /permissions revoke <tool>
             Revoke a persistent permission rule.
  /permissions clear
             Clear session-level permission rules.
  /tool NAME INPUT
             Run one registered tool directly.
  /tool list
             Show enabled and available tools by group.
  /exit      Exit the REPL.

Press Shift+Enter for a newline. If your terminal does not send Shift+Enter,
use Esc Enter as the fallback accepted by prompt_toolkit.
Use Tab to complete slash commands, /tool names, and @file references.
"""


@dataclass
class ReplState:
    mode: str = "act"
    verbose: bool = False
    approved_plan: str | None = None
    exit_pending: float = 0.0


class ReplHITLHandler:
    def __init__(
        self,
        session_policy: SessionPermissionPolicy,
        persistent_store: PersistentPermissionStore,
        prompt: PromptLike | None = None,
    ) -> None:
        self.session_policy = session_policy
        self.persistent_store = persistent_store
        self.prompt = prompt

    def __call__(self, tool: ToolSpec, action_input: str) -> HITLResult:
        session_decision = self.session_policy.decide(tool.name, action_input)
        if session_decision is not None and session_decision != "ask":
            return HITLResult(session_decision, "session")
        persistent_policy = self.persistent_store.load()
        pers_decision = persistent_policy.decide(tool.name, action_input)
        if pers_decision is not None and pers_decision != "ask":
            return HITLResult(pers_decision, "permanent")
        if _should_use_radiolist(self.prompt):
            choice = _radiolist_prompt(tool, action_input)
        elif self.prompt is not None and not _is_prompt_toolkit_prompt(self.prompt):
            choice = self.prompt.prompt(self._prompt_text(tool, action_input)).strip()
        else:
            choice = input(self._terminal_prompt_text(tool, action_input)).strip()
        return self._apply_choice(choice, tool, action_input)

    def _prompt_text(self, tool: ToolSpec, action_input: str) -> str:
        brief = _brief_input(tool.name, action_input)
        return (
            f"需要授权：{tool.name}"
            f"\n  指令：{brief}"
            f"\n  风险：{tool.risk}"
            f"\n  选项："
            f"\n    1) 允许（仅本次）"
            f"\n    2) 此次对话中允许"
            f"\n    3) 始终允许"
            f"\n    4) 拒绝"
        )

    def _terminal_prompt_text(self, tool: ToolSpec, action_input: str) -> str:
        return f"\r\033[K\n{self._prompt_text(tool, action_input)}\napprove [1-4]> "

    def _apply_choice(
        self, choice: str, tool: ToolSpec, action_input: str
    ) -> HITLResult:
        if choice == "1":
            return HITLResult("allow", "once")
        if choice == "2":
            self.session_policy.grant(tool.name, "allow", action_input)
            return HITLResult("allow", "session")
        if choice == "3":
            self.persistent_store.grant(tool.name, "allow", action_input)
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once")


def _has_radiolist() -> bool:
    try:
        from prompt_toolkit.shortcuts.dialogs import radiolist_dialog  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_radiolist(prompt: PromptLike | None) -> bool:
    if _is_async_loop_running():
        return False
    if prompt is not None and _is_prompt_toolkit_prompt(prompt):
        return False
    return _has_radiolist()


def _is_async_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _is_prompt_toolkit_prompt(prompt: PromptLike | None) -> bool:
    if prompt is None:
        return False
    module = type(prompt).__module__
    return module.startswith("prompt_toolkit.")


def _radiolist_prompt(tool: ToolSpec, action_input: str) -> str:
    from prompt_toolkit.shortcuts.dialogs import radiolist_dialog

    brief = _brief_input(tool.name, action_input)
    result = radiolist_dialog(
        title=f"需要授权：{tool.name}",
        text=f"指令：{brief}    风险：{tool.risk}",
        values=[
            ("1", "允许（仅本次）"),
            ("2", "此次对话中允许"),
            ("3", "始终允许"),
            ("4", "拒绝"),
        ],
        default=None,
    ).run()
    return result or "4"


def run_repl(
    app,
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
    session = prompt_session or create_prompt_session(root, registry)
    state = ReplState()
    session_policy = SessionPermissionPolicy()
    persistent_store = PersistentPermissionStore(root / ".local" / "hitl_policy.json")
    hitl_handler = ReplHITLHandler(session_policy, persistent_store, session)
    agent = getattr(app, "agent", None)
    if agent is not None:
        agent.approval_callback = hitl_handler
    print("Xcode REPL")
    print("Type /help for commands.")
    if resume_latest:
        _resume_interactively(store, session)

    while True:
        try:
            text = session.prompt("xcode> ").strip()
        except (EOFError, KeyboardInterrupt):
            now = time.time()
            if state.exit_pending and now - state.exit_pending < 1.5:
                print()
                _print_saved_conversation(store)
                return 0
            state.exit_pending = now
            print()
            sys.stdout.write("\033[90m(press Ctrl+C again to exit)\033[0m\n")
            sys.stdout.flush()
            continue
        if not text:
            continue
        if text.startswith("/"):
            if _handle_command(
                text,
                store,
                app,
                markdown_renderer,
                state,
                session,
                session_policy,
                persistent_store,
            ):
                _print_saved_conversation(store)
                return 0
            continue

        store.append("user", text)
        expanded_text, references = expand_file_references(text, root)
        if references:
            store.append("event", _file_reference_event(references))
        agent_text = expanded_text
        if state.approved_plan is not None:
            agent_text = f"<approved-plan>\n{state.approved_plan}\n</approved-plan>\n{expanded_text}"
            state.approved_plan = None
        step_answers: list[str] = []
        current_step_thoughts: list[str] = []
        stopped_reason: str | None = None
        interrupted = False
        live_console = Console(file=sys.stdout)

        def _safe_write(text: str) -> None:
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

        def _clear_line() -> None:
            _safe_write("\r\033[K")

        reasoning_started_at: float | None = None
        reasoning_text = ""
        reasoning_preview = _LiveReasoningPreview(live_console)
        answer_stream = _LiveMarkdownStream(live_console)
        streamed_text = False
        tool_group: dict[str, Any] | None = None
        tool_call_labels: dict[str, str] = {}

        def _record_tool_call(event_data: Any) -> None:
            nonlocal tool_group
            label = _brief_input(event_data.name, event_data.input)
            intent = _tool_intent(event_data.name, event_data.input)
            tool_call_labels[event_data.id] = label
            if state.verbose:
                _print_tool_call_rich(label, live_console)
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

        def _record_tool_result(event_data: Any) -> None:
            if state.verbose:
                _print_tool_result_rich(event_data, state.verbose, live_console)
                return
            if tool_group is None:
                return
            if event_data.status == "ok":
                tool_group["ok"] += 1
                return
            label = tool_call_labels.get(event_data.tool_use_id, event_data.tool_use_id)
            tool_group["errors"].append((label, event_data))

        def _flush_tool_group() -> None:
            nonlocal tool_group
            if tool_group is None:
                return
            calls = int(tool_group["calls"])
            errors = list(tool_group["errors"])
            intents = list(tool_group["intents"])
            title = _summarize_intents(intents)
            status = "failed" if errors else "done"
            style = "red" if errors else "green"
            live_console.print(Text(f"  • Explore: {title}", style="yellow"))
            live_console.print(Text(f"    {status}: {calls} tools", style=style))
            for label, result in errors:
                summary = _single_line_preview(str(result.content), width=120)
                live_console.print(Text(f"    error: {label}: {summary}", style="red"))
            tool_group = None

        def _render_reasoning_preview() -> None:
            lines = _reasoning_preview_lines(reasoning_text)
            if lines:
                reasoning_preview.update(lines)

        def _finish_reasoning_preview() -> None:
            nonlocal reasoning_started_at, reasoning_text
            if reasoning_started_at is None:
                return
            elapsed = time.perf_counter() - reasoning_started_at
            if reasoning_text and elapsed < _MIN_REASONING_PREVIEW_SECONDS:
                time.sleep(_MIN_REASONING_PREVIEW_SECONDS - elapsed)
                elapsed = time.perf_counter() - reasoning_started_at
            reasoning_preview.stop()
            preview = _single_line_preview(reasoning_text)
            live_console.print(
                Text(
                    f"  • thinked for {_format_elapsed(elapsed)}", style=_THINKING_STYLE
                )
            )
            if preview:
                live_console.print(Text(f"    {preview}", style=_THINKING_STYLE))
            reasoning_started_at = None
            reasoning_text = ""

        try:
            for event in _ask_stream(app, agent_text, state.mode):
                store.append("event", _event_to_dict(event))
                if event.type in {"reasoning_delta", "thinking_delta"}:
                    _flush_tool_group()
                    if reasoning_started_at is None:
                        reasoning_started_at = time.perf_counter()
                    reasoning_text += str(event.data)
                    _render_reasoning_preview()
                    continue

                _finish_reasoning_preview()

                if event.type == "text_delta":
                    _flush_tool_group()
                    chunk = str(event.data)
                    current_step_thoughts.append(chunk)
                    streamed_text = True
                    answer_stream.update("".join(current_step_thoughts))
                elif event.type == "assistant":
                    has_tool_calls = False
                    if isinstance(event.data, list):
                        has_tool_calls = any(
                            isinstance(block, dict) and block.get("type") == "tool_use"
                            for block in event.data
                        )
                    if has_tool_calls:
                        thoughts = "".join(current_step_thoughts).strip()
                        if thoughts:
                            answer_stream.stop()
                            if step_answers:
                                print()
                            step_answers.append(thoughts)
                        current_step_thoughts = []
                        streamed_text = False
                elif event.type == "tool_use":
                    _record_tool_call(event.data)
                elif event.type == "tool_result":
                    _record_tool_result(event.data)
                elif event.type == "final":
                    _flush_tool_group()
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
                    stopped_reason = _final_stop_reason(event.data)
        except KeyboardInterrupt:
            interrupted = True
            token = getattr(getattr(app, "agent", None), "cancellation_token", None)
            if token is not None:
                token.cancel("interrupted by user")
            tool_group = None
            _clear_line()
            store.append(
                "event", {"type": "interrupted", "data": "interrupted by user"}
            )
            print("[interrupted] current run cancelled; session is still active.")
            continue
        finally:
            _finish_reasoning_preview()
            answer_stream.stop()
            if not interrupted:
                _flush_tool_group()

        if stopped_reason:
            print(stopped_reason)
        if step_answers:
            answer = "\n\n".join(step_answers)
            store.append("assistant", answer)
            store.update_summary()


def create_prompt_session(
    project_root: Path | None = None,
    registry: tuple[ToolSpec, ...] = (),
) -> PromptLike:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError as exc:
        raise RuntimeError(
            "prompt_toolkit is required for REPL mode. Install it in .venv first."
        ) from exc

    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event) -> None:
        event.current_buffer.validate_and_handle()

    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    try:
        bindings.add("s-enter")(insert_newline)
    except ValueError:
        pass
    bindings.add("escape", "enter")(insert_newline)

    completer = ReplCompleter(project_root or Path.cwd(), registry)

    history = None
    try:
        from prompt_toolkit.history import FileHistory

        history_dir = (project_root or Path.cwd()) / ".local"
        history_dir.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_dir / "repl_history"))
    except OSError:
        pass

    return PromptSession(
        multiline=True,
        key_bindings=bindings,
        completer=completer,
        complete_while_typing=True,
        history=history,
    )


def _handle_permissions(
    command: str,
    session_policy: SessionPermissionPolicy | None,
    persistent_store: PersistentPermissionStore | None,
) -> None:
    parts = command.split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else ""
    if sub == "revoke" and len(parts) >= 3 and persistent_store is not None:
        tool_name = parts[2]
        persistent_store.revoke(tool_name)
        print(f"Revoked persistent permission for: {tool_name}")
        return
    if sub == "clear" and session_policy is not None:
        session_policy.clear()
        print("Session permissions cleared.")
        return
    _list_permissions(session_policy, persistent_store)


def _list_permissions(
    session_policy: SessionPermissionPolicy | None,
    persistent_store: PersistentPermissionStore | None,
) -> None:
    lines = ["<permissions>"]
    if session_policy is not None:
        rules = list(session_policy._rules)
        if rules:
            lines.append("  session:")
            for r in rules:
                ic = f" (input: {r.input_contains})" if r.input_contains else ""
                lines.append(f"    {r.tool} = {r.decision}{ic}")
    if persistent_store is not None:
        policy = persistent_store.load()
        if policy.rules:
            lines.append("  persistent:")
            for r in policy.rules:
                ic = f" (input: {r.input_contains})" if r.input_contains else ""
                lines.append(f"    {r.tool} = {r.decision}{ic}")
    if len(lines) == 1:
        lines.append("  (none)")
    lines.append("</permissions>")
    print("\n".join(lines))


def _handle_model_command(command: str, app) -> None:
    parts = command.split(maxsplit=3)
    if len(parts) == 1:
        info = app.get_model_info() if hasattr(app, "get_model_info") else {}
        if info:
            print(f"  Model    : {info.get('model', 'unknown')}")
            print(f"  Base URL : {info.get('base_url', '')}")
        else:
            print("Model info not available.")
        return

    model_name = parts[1]
    kwargs: dict[str, object] = {"model": model_name}

    # 解析 --thinking <level> 选项
    if len(parts) >= 4 and parts[2] == "--thinking":
        level = parts[3].lower()
        if level not in ("off", "minimal", "low", "medium", "high", "xhigh"):
            print(
                f"Invalid thinking level: {level}. Use off/minimal/low/medium/high/xhigh."
            )
            return
        if level == "off":
            kwargs["thinking"] = False
            kwargs["reasoning_effort"] = None
        else:
            kwargs["thinking"] = True
            kwargs["reasoning_effort"] = level

    if not hasattr(app, "set_model"):
        print("Model switching is not supported in this app.")
        return

    try:
        new_model = app.set_model(**kwargs)
        print(f"Switched to model: {new_model}")
    except Exception as e:
        print(f"Failed to switch model: {e}")


def _handle_effort_command(command: str, app) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        info = app.get_model_info() if hasattr(app, "get_model_info") else {}
        current = info.get("reasoning_effort", "not set") if info else "unknown"
        print(f"  Reasoning effort: {current}")
        return

    level = parts[1].lower()
    if level not in ("off", "minimal", "low", "medium", "high", "max"):
        print("Invalid effort level. Use: off/minimal/low/medium/high/max")
        return

    if not hasattr(app, "set_model"):
        print("Model switching is not supported in this app.")
        return

    info = app.get_model_info() if hasattr(app, "get_model_info") else {}
    current_model = info.get("model", "unknown") if info else "unknown"

    try:
        if level == "off":
            app.set_model(model=current_model, thinking=False, reasoning_effort=None)
            print("Reasoning effort disabled.")
        else:
            app.set_model(model=current_model, thinking=True, reasoning_effort=level)
            print(f"Reasoning effort set to: {level}")
    except Exception as e:
        print(f"Failed to set reasoning effort: {e}")


def _handle_command(
    command: str,
    store: SessionStore,
    app,
    renderer: MarkdownRenderer,
    state: ReplState,
    prompt_session: PromptLike,
    session_policy: SessionPermissionPolicy | None = None,
    persistent_store: PersistentPermissionStore | None = None,
) -> bool:
    if command in {"/exit", "/quit"}:
        return True
    if command == "/help":
        print(HELP_TEXT)
        return False
    if command == "/clear":
        store.clear()
        if session_policy is not None:
            session_policy.clear()
        print("New session started.")
        return False
    if command == "/fork" or command.startswith("/fork "):
        parts = command.split(maxsplit=1)
        fork_type = parts[1].strip() if len(parts) == 2 else None
        if fork_type is not None and fork_type not in FORK_TYPES:
            print(f"fork_type must be one of {sorted(FORK_TYPES)}, got {fork_type!r}")
            return False
        meta = store.fork_into(fork_type)
        if session_policy is not None:
            session_policy.clear()
        label = f" ({fork_type})" if fork_type else ""
        print(f'Forked: "{meta.title}"{label}')
        return False
    if command.startswith("/rewind"):
        parts = command.split()
        turns = int(parts[1]) if len(parts) > 1 else 1
        removed = store.rewind_turns(turns)
        print(f"Rewound {removed} transcript records.")
        return False
    if command.startswith("/resume"):
        parts = command.split(maxsplit=1)
        if len(parts) == 2:
            target = parts[1].strip()
            if target == "last":
                view = _resume_latest(store)
                if view:
                    print(_resumed_message(view))
                    _print_loaded_history(store)
                else:
                    print("No conversations found.")
                return False
            store.resume(target)
            print(_resumed_message(_current_view(store)))
            _print_loaded_history(store)
            return False
        _resume_interactively(store, prompt_session)
        return False
    if command == "/sessions":
        _print_sessions(store.list_session_infos())
        return False
    if command == "/model" or command.startswith("/model "):
        _handle_model_command(command, app)
        return False
    if command == "/effort" or command.startswith("/effort "):
        _handle_effort_command(command, app)
        return False
    if command == "/plan":
        state.mode = "plan"
        print(
            "Plan Mode enabled. Read-only inspection tools are available; edits and shell are blocked."
        )
        return False
    if command == "/review":
        state.mode = "review"
        print("Review Mode enabled. Edits are blocked; validation requires approval.")
        return False
    if command == "/act" or command.startswith("/act "):
        is_clear = False
        parts = command.split(maxsplit=1)
        if len(parts) == 2 and parts[1].strip() == "--clear":
            is_clear = True

        choice = "1" if is_clear else None
        if choice is None:
            print("\nSelect action:")
            print("  1) Clear and Act (Clear context, keep plan, and act)")
            print("  2) Keep and Act (Keep current context and act directly)")
            print("  3) Review Mode")
            print("  4) Continue in Plan Mode")
            try:
                choice = prompt_session.prompt("Choice (1-4): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return False

        if choice == "1":
            records = store.load_records()
            last_assistant_content = None
            for r in reversed(records):
                if r.type == "assistant":
                    last_assistant_content = r.content
                    break

            if not last_assistant_content or not str(last_assistant_content).strip():
                print(
                    "Error: No plan found in the last assistant reply. Cannot Clear and Act."
                )
                return False

            parent_id = store.current_path.stem.removeprefix("session-")
            from datetime import datetime

            plan_text = f"# Approved Plan (Forked from {parent_id})\nDate: {datetime.now().isoformat(timespec='seconds')}\n\n{last_assistant_content}"

            plan_file = store.artifacts_dir / f"plan-{parent_id}.md"
            try:
                plan_file.write_text(plan_text, encoding="utf-8")
            except OSError as e:
                print(f"Warning: Failed to write plan artifact: {e}")

            meta = store.fork_clean_into(
                "isolate", title=f"Act Continuation of Plan {parent_id}"
            )
            if session_policy is not None:
                session_policy.clear()

            state.approved_plan = str(last_assistant_content)
            state.mode = "act"
            print(f'Clean Fork created: "{meta.title}"')
            print("Act Mode enabled with approved plan.")
            return False

        elif choice == "2":
            state.mode = "act"
            print("Act Mode enabled. Normal tool use restored within policy.")
            return False
        elif choice == "3":
            state.mode = "review"
            print(
                "Review Mode enabled. Edits are blocked; validation requires approval."
            )
            return False
        elif choice == "4":
            print("Continuing in Plan Mode.")
            return False
        else:
            print(f"Invalid choice: {choice}")
            return False
    if command == "/verbose on":
        state.verbose = True
        print("Verbose mode on: tool call ids and result details will be shown.")
        return False
    if command == "/verbose off":
        state.verbose = False
        print("Verbose mode off.")
        return False
    if command == "/compact":
        agent = getattr(app, "agent", None)
        if agent is not None and hasattr(agent, "request_compaction"):
            agent.request_compaction()
            print("Active context compaction requested for the next agent run.")
        else:
            print(
                "Context compaction is not supported or not configured in the current agent."
            )
        compacted = store.compact_current_session(max_tool_result_chars=200)
        if compacted > 0:
            print(f"Compacted {compacted} large tool results in the session log.")
        else:
            print("No large tool results to compact in the session log.")
        return False
    if command == "/permissions" or command.startswith("/permissions "):
        _handle_permissions(command, session_policy, persistent_store)
        return False
    if command == "/tool" or command.startswith("/tool "):
        output = _run_tool_command(command, app)
        store.append("event", {"type": "tool_command", "data": command})
        store.append("event", {"type": "tool_result", "data": output})
        renderer.render(output)
        return False
    print(f"Unknown command: {command}")
    return False


def _resume_interactively(store: SessionStore, prompt_session: PromptLike) -> None:
    sessions = store.list_session_infos()
    if not sessions:
        print("No conversations found.")
        return
    _print_sessions(sessions)
    choice = prompt_session.prompt("resume> ").strip()
    if not choice:
        print("Resume cancelled.")
        return
    selected = _select_session(sessions, choice)
    if selected is None:
        print(f"No conversation matched: {choice}")
        return
    store.resume(selected.id)
    print(_resumed_message(selected))
    _print_loaded_history(store)


def _resume_latest(store: SessionStore) -> SessionMetadataView | None:
    sessions = store.list_session_infos(limit=1)
    if not sessions:
        return None
    store.resume(sessions[0].id)
    return sessions[0]


def _select_session(
    sessions: list[SessionMetadataView],
    choice: str,
) -> SessionMetadataView | None:
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
    for item in sessions:
        if choice in {item.id, item.title}:
            return item
    return None


def _print_sessions(sessions: list[SessionMetadataView]) -> None:
    id_to_index = {s.id: str(i) for i, s in enumerate(sessions, 1)}
    for index, item in enumerate(sessions, start=1):
        suffix = ""
        if item.parent_id and item.parent_id in id_to_index:
            suffix = f" (forked from #{id_to_index[item.parent_id]})"
        print(f"{index}. {item.title}{suffix}")
        if item.summary:
            print(f"   {item.summary}")


def _current_view(store: SessionStore) -> SessionMetadataView:
    metadata = store.current_metadata()
    if metadata is not None:
        return SessionMetadataView(
            id=metadata.id,
            title=metadata.title,
            summary=metadata.summary,
            updated_at=metadata.updated_at,
            path=store.current_path,
        )
    session_id = store.current_path.stem.removeprefix("session-")
    return SessionMetadataView(
        id=session_id,
        title=f"Session {session_id}",
        summary="No summary available.",
        updated_at="",
        path=store.current_path,
    )


def _resumed_message(view: SessionMetadataView) -> str:
    return f"Resumed conversation: {view.title}"


def _print_loaded_history(store: SessionStore) -> None:
    console = Console(file=sys.stdout)
    records = [
        record
        for record in store.load_records()
        if record.type in {"user", "assistant"} and str(record.content).strip()
    ]
    if not records:
        return
    console.print(
        Text(
            f"  • loaded {len(records)} message(s) from this session branch",
            style="blue",
        )
    )
    for record in records:
        if record.type == "assistant":
            console.print(Text("assistant:", style="green"))
            console.print(Markdown(str(record.content)))
        else:
            console.print(Text(f"user: {record.content}", style="cyan"))


def _print_saved_conversation(store: SessionStore) -> None:
    metadata = store.update_summary()
    if metadata is not None:
        print(f"Conversation saved: {metadata.title}")


def _ask_stream(app, text: str, mode: str):
    return app.ask_stream(text, mode=mode)


from xcode.cli.tool_catalog import build_tool_catalog  # noqa: E402


def _run_tool_command(command: str, app) -> str:
    parts = command.split(maxsplit=2)
    if len(parts) < 2:
        return "usage: /tool NAME INPUT\n/tool list — show enabled tools by group"
    name = parts[1]
    registry: tuple[ToolSpec, ...] = tuple(getattr(app, "registry", ()) or ())
    if name == "list":
        catalog = build_tool_catalog()
        enabled_names = {t.name for t in registry}

        lines = ["<visible tools>"]
        core_names = sorted(t.name for t in registry if t.group == "core")
        if core_names:
            lines.append("  core:")
            lines.extend(f"    {n}" for n in core_names)

        noncore_groups = sorted({t.group for t in registry if t.group != "core"})
        for g in noncore_groups:
            lines.append(f"  {g}:")
            tools_in_group = sorted(
                (t for t in registry if t.group == g), key=lambda x: x.name
            )
            for t in tools_in_group:
                suffix = ""
                if t.group == "mcp":
                    if "[mcp: " in t.description:
                        server_name = t.description.split("[mcp: ")[-1].split("]")[0]
                        suffix = f" [mcp: {server_name}]"
                lines.append(f"    {t.name}{suffix}")
        lines.append("</visible tools>")

        all_known = set()
        for group_names in catalog.values():
            all_known.update(group_names)
        hidden = sorted(all_known - enabled_names)
        if hidden:
            lines.append("<hidden tools (enable via tools.enabled_groups)>")
            for g in sorted(catalog):
                group_hidden = sorted(catalog[g] & set(hidden))
                if group_hidden:
                    lines.append(f"  {g}:")
                    lines.extend(f"    {n}" for n in group_hidden)
            lines.append("</hidden tools>")

        available_groups = sorted(catalog.keys() - {t.group for t in registry})
        if available_groups:
            lines.append("<available groups>")
            lines.extend(f"  {g}" for g in available_groups)
            lines.append("</available groups>")
        return "\n".join(lines)
    action_input = parts[2] if len(parts) == 3 else ""
    result = run_tool_result(
        {tool.name: tool for tool in registry},
        name,
        action_input,
    )
    return result.content


def _brief_input(name: str, raw_input: Any) -> str:
    """从工具输入中提取简短的人类可读摘要。"""
    if isinstance(raw_input, dict):
        if name == "bash":
            command = raw_input.get("command") or raw_input.get("input")
            return _single_line_preview(f"bash: {command}") if command else name
        parts = []
        for key, value in raw_input.items():
            if value in (None, "", [], {}):
                continue
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
        if parts:
            return _single_line_preview(f"{name}: {', '.join(parts)}")
        if raw_input:
            key, val = next(iter(raw_input.items()))
            return _single_line_preview(f"{name}: {key}={val}")
        return name
    if isinstance(raw_input, str) and raw_input:
        return _single_line_preview(f"{name}: {raw_input}")
    return name


def _tool_intent(name: str, raw_input: Any) -> str:
    if not isinstance(raw_input, dict):
        return _single_line_preview(f"Run {name}")
    if name == "grep_search":
        pattern = (
            raw_input.get("pattern") or raw_input.get("query") or raw_input.get("input")
        )
        path = raw_input.get("path") or raw_input.get("include") or "workspace"
        if pattern:
            return _single_line_preview(f"Search {path} for {pattern}")
    if name == "glob_files":
        pattern = (
            raw_input.get("pattern") or raw_input.get("path") or raw_input.get("input")
        )
        path = raw_input.get("path") if raw_input.get("pattern") else "workspace"
        if pattern:
            return _single_line_preview(f"Find {pattern} in {path}")
    if name == "read_file":
        path = raw_input.get("path") or raw_input.get("input")
        if path:
            return _single_line_preview(f"Read {path}")
    if name in {"write_file", "edit_file"}:
        path = raw_input.get("path") or raw_input.get("input")
        if path:
            return _single_line_preview(f"Edit {path}")
    if name == "bash":
        command = raw_input.get("command") or raw_input.get("input")
        if command:
            return _single_line_preview(f"Run {command}")
    return _single_line_preview(f"Run {name}")


def _summarize_intents(intents: list[str]) -> str:
    if not intents:
        return "workspace"
    if len(intents) == 1:
        return intents[0]
    first = intents[0]
    return _single_line_preview(f"{first} and {len(intents) - 1} more")


def _event_to_dict(event) -> dict[str, Any]:
    data = event.data
    if is_dataclass(data) and not isinstance(data, type):
        payload = asdict(data)
    else:
        payload = data
    return {"type": event.type, "step": event.step, "data": payload}


def _print_tool_call_rich(label: str, console: Console | None = None) -> None:
    target = console or _console
    target.print(Text(f"  • {label}", style="yellow"))


def _print_tool_result_rich(
    data,
    verbose: bool,
    console: Console | None = None,
) -> None:
    if data.status == "ok" and not verbose:
        return
    target = console or _console
    border = {
        "error": "red",
        "denied": "red",
        "approval_required": "yellow",
    }.get(data.status, "green" if data.status == "ok" else "cyan")
    mark = {"error": "✘", "denied": "⊘", "approval_required": "?"}.get(
        data.status, data.status
    )
    limit = 600 if verbose else 200
    summary = _single_line_preview(str(data.content), width=limit)
    target.print(Text(f"  ← {mark} {summary}", style=border))


def _final_stop_reason(data) -> str | None:
    if getattr(data, "stopped_by_limit", False):
        return "[stopped] step limit reached"
    if getattr(data, "stopped_by_watchdog", False):
        reason = getattr(data, "watchdog_reason", "repeated tool calls detected")
        return f"[stopped] {reason}"
    return None


def _file_reference_event(references: list[FileReference]) -> dict[str, Any]:
    return {
        "type": "file_references",
        "data": [
            {
                "path": reference.path,
                "status": reference.status,
                "error": reference.error,
            }
            for reference in references
        ],
    }
