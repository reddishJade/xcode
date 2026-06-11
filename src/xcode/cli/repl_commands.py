from __future__ import annotations

from datetime import datetime

from .commands import (
    CommandContext,
    CommandEntry,
    command_names,
    generate_help_text,
)
from .repl_sessions import (
    current_view,
    print_loaded_history,
    resume_interactively,
    resume_latest,
    resumed_message,
    sync_agent_history,
)
from .repl_settings import (
    handle_effort_command,
    handle_model_command,
    handle_permissions,
    handle_thinking_command,
)
from .repl_tools import run_tool_command
from xcode.harness.observability import (
    PersistentPermissionStore,
    SessionPermissionPolicy,
)
from xcode.harness.session import FORK_TYPES, SessionStore


def cmd_help(cmd: str, ctx: CommandContext) -> bool:
    print(HELP_TEXT)
    return False


def cmd_clear(cmd: str, ctx: CommandContext) -> bool:
    ctx.store.clear()
    if ctx.session_policy is not None:
        ctx.session_policy.clear()
    sync_agent_history(ctx.app, ctx.store)
    print("New session started.")
    return False


def cmd_fork(cmd: str, ctx: CommandContext) -> bool:
    parts = cmd.split(maxsplit=1)
    fork_type = parts[1].strip() if len(parts) == 2 else None
    if fork_type is not None and fork_type not in FORK_TYPES:
        print(f"fork_type must be one of {sorted(FORK_TYPES)}, got {fork_type!r}")
        return False
    meta = ctx.store.fork_into(fork_type)
    if ctx.session_policy is not None:
        ctx.session_policy.clear()
    sync_agent_history(ctx.app, ctx.store)
    label = f" ({fork_type})" if fork_type else ""
    print(f'Forked: "{meta.title}"{label}')
    return False


def cmd_rewind(cmd: str, ctx: CommandContext) -> bool:
    parts = cmd.split()
    turns = int(parts[1]) if len(parts) > 1 else 1
    removed = ctx.store.rewind_turns(turns)
    sync_agent_history(ctx.app, ctx.store)
    print(f"Rewound {removed} transcript records.")
    return False


def cmd_resume(cmd: str, ctx: CommandContext) -> bool:
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2:
        target = parts[1].strip()
        if target == "last":
            view = resume_latest(ctx.store)
            if view:
                print(resumed_message(view))
                print_loaded_history(ctx.store)
                sync_agent_history(ctx.app, ctx.store)
            else:
                print("No conversations found.")
            return False
        ctx.store.resume(target)
        print(resumed_message(current_view(ctx.store)))
        print_loaded_history(ctx.store)
        sync_agent_history(ctx.app, ctx.store)
        return False
    resume_interactively(ctx.store, ctx.prompt_session)
    sync_agent_history(ctx.app, ctx.store)
    return False


def cmd_tree(cmd: str, ctx: CommandContext) -> bool:
    nodes = ctx.store.get_tree()
    if not nodes:
        print("No session tree available (no metadata).")
        return False

    for node in nodes:
        indent = "  " * node.depth
        prefix = "└─ " if node.depth > 0 else ""
        branch = "🌿 " if node.fork_type else "  "
        marker = " ← current" if node.is_current else ""
        label = f"{branch}{node.title}"
        print(f"{indent}{prefix}{label}{marker}")
    return False


def cmd_branch(cmd: str, ctx: CommandContext) -> bool:
    parts = cmd.split(maxsplit=1)
    if len(parts) == 1 or parts[1].strip() in {"list", "tree"}:
        return cmd_tree("/tree", ctx)

    target = parts[1].strip()
    try:
        view = ctx.store.switch_branch(target)
    except ValueError as exc:
        print(str(exc))
        return False
    if ctx.session_policy is not None:
        ctx.session_policy.clear()
    sync_agent_history(ctx.app, ctx.store)
    print(resumed_message(view))
    print_loaded_history(ctx.store)
    return False


def cmd_sessions(cmd: str, ctx: CommandContext) -> bool:
    import questionary

    sessions = ctx.store.list_session_infos()
    if not sessions:
        print("No conversations found.")
        return False

    id_to_index = {session.id: str(i) for i, session in enumerate(sessions, 1)}
    choices = []
    for i, item in enumerate(sessions, 1):
        title = f"{i}. {item.title}"
        if item.parent_id and item.parent_id in id_to_index:
            title += f" (forked from #{id_to_index[item.parent_id]})"
        if item.summary:
            title += f" — {item.summary}"
        choices.append(questionary.Choice(title=title[:120], value=item))

    selected = questionary.select("Select session to resume:", choices=choices).ask()
    if selected is None:
        return False

    ctx.store.resume(selected.id)
    sync_agent_history(ctx.app, ctx.store)
    print(resumed_message(selected))
    print_loaded_history(ctx.store)
    return False


def cmd_model(cmd: str, ctx: CommandContext) -> bool:
    handle_model_command(cmd, ctx.app)
    return False


def cmd_effort(cmd: str, ctx: CommandContext) -> bool:
    handle_effort_command(cmd, ctx.app)
    return False


def cmd_thinking(cmd: str, ctx: CommandContext) -> bool:
    handle_thinking_command(cmd, ctx.app)
    return False


def cmd_plan(cmd: str, ctx: CommandContext) -> bool:
    ctx.state.mode = "plan"
    print(
        "Plan Mode enabled. Read-only inspection tools are available; edits and shell are blocked."
    )
    return False


def cmd_review(cmd: str, ctx: CommandContext) -> bool:
    ctx.state.mode = "review"
    print("Review Mode enabled. Edits are blocked; validation requires approval.")
    return False


def cmd_act(cmd: str, ctx: CommandContext) -> bool:
    is_clear = False
    parts = cmd.split(maxsplit=1)
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
            choice = ctx.prompt_session.prompt("Choice (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return False

    if choice == "1":
        records = ctx.store.load_records()
        last_assistant_content = None
        for record in reversed(records):
            if record.type == "assistant":
                last_assistant_content = record.content
                break

        if not last_assistant_content or not str(last_assistant_content).strip():
            print(
                "Error: No plan found in the last assistant reply. Cannot Clear and Act."
            )
            return False

        parent_id = ctx.store.current_path.stem.removeprefix("session-")
        plan_text = (
            f"# Approved Plan (Forked from {parent_id})\n"
            f"Date: {datetime.now().isoformat(timespec='seconds')}\n\n"
            f"{last_assistant_content}"
        )

        plan_file = ctx.store.artifacts_dir / f"plan-{parent_id}.md"
        try:
            plan_file.write_text(plan_text, encoding="utf-8")
        except OSError as exc:
            print(f"Warning: Failed to write plan artifact: {exc}")

        meta = ctx.store.fork_clean_into(
            "isolate", title=f"Act Continuation of Plan {parent_id}"
        )
        if ctx.session_policy is not None:
            ctx.session_policy.clear()
        sync_agent_history(ctx.app, ctx.store)

        ctx.state.approved_plan = str(last_assistant_content)
        ctx.state.mode = "act"
        print(f'Clean Fork created: "{meta.title}"')
        print("Act Mode enabled with approved plan.")
    elif choice == "2":
        ctx.state.mode = "act"
        print("Act Mode enabled. Normal tool use restored within policy.")
    elif choice == "3":
        ctx.state.mode = "review"
        print("Review Mode enabled. Edits are blocked; validation requires approval.")
    elif choice == "4":
        print("Continuing in Plan Mode.")
    else:
        print(f"Invalid choice: {choice}")
    return False


def cmd_verbose(cmd: str, ctx: CommandContext) -> bool:
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2 and parts[1] == "on":
        ctx.state.verbose = True
        print("Verbose mode on: tool call ids and result details will be shown.")
    elif len(parts) == 2 and parts[1] == "off":
        ctx.state.verbose = False
        print("Verbose mode off.")
    else:
        print("Usage: /verbose on|off")
    return False


def cmd_queue(cmd: str, ctx: CommandContext) -> bool:
    parts = cmd.split(maxsplit=1)
    if len(parts) == 1:
        state = "on" if ctx.state.queue_mode else "off"
        print(f"Queue mode: {state}")
        return False
    value = parts[1].strip().lower()
    if value not in {"on", "off"}:
        print("Usage: /queue on|off")
        return False
    ctx.state.queue_mode = value == "on"
    print(f"Queue mode {'enabled' if ctx.state.queue_mode else 'disabled'}.")
    return False


def cmd_compact(cmd: str, ctx: CommandContext) -> bool:
    agent = getattr(ctx.app, "agent", None)
    if agent is not None and hasattr(agent, "request_compaction"):
        agent.request_compaction()
        print("Active context compaction requested for the next agent run.")
    else:
        print(
            "Context compaction is not supported or not configured in the current agent."
        )
    compacted = ctx.store.compact_current_session(max_tool_result_chars=200)
    if compacted > 0:
        print(f"Compacted {compacted} large tool results in the session log.")
    else:
        print("No large tool results to compact in the session log.")
    return False


def cmd_permissions(cmd: str, ctx: CommandContext) -> bool:
    handle_permissions(cmd, ctx.session_policy, ctx.persistent_store)
    return False


def cmd_tool(cmd: str, ctx: CommandContext) -> bool:
    output = run_tool_command(cmd, ctx.app)
    ctx.store.append("event", {"type": "tool_command", "data": cmd})
    ctx.store.append("event", {"type": "tool_result", "data": output})
    ctx.renderer.render(output)
    return False


def cmd_exit(cmd: str, ctx: CommandContext) -> bool:
    return True


COMMAND_REGISTRY: dict[str, CommandEntry] = {
    "/help": CommandEntry(handler=cmd_help, desc="Show this help."),
    "/clear": CommandEntry(handler=cmd_clear, desc="Start a new session transcript."),
    "/fork": CommandEntry(
        handler=cmd_fork,
        desc="Fork current session into an independent branch.",
        args_desc="[explore|verify|isolate]",
        accepts_args=True,
    ),
    "/rewind": CommandEntry(
        handler=cmd_rewind,
        desc="Remove the last N user turns from the transcript.",
        args_desc="N",
        accepts_args=True,
    ),
    "/resume": CommandEntry(
        handler=cmd_resume,
        desc="Choose a recent conversation to resume.",
        accepts_args=True,
    ),
    "/sessions": CommandEntry(handler=cmd_sessions, desc="List recent conversations."),
    "/tree": CommandEntry(
        handler=cmd_tree,
        desc="Show session fork tree.",
    ),
    "/branch": CommandEntry(
        handler=cmd_branch,
        desc="List or switch session branches.",
        args_desc="list|tree|<id|title>",
        accepts_args=True,
    ),
    "/model": CommandEntry(
        handler=cmd_model,
        desc="Show current model info.",
        args_desc="[profile/]name[:thinking] [--thinking <level>]",
        accepts_args=True,
    ),
    "/effort": CommandEntry(
        handler=cmd_effort,
        desc="Show current reasoning effort.",
        args_desc="<off|minimal|low|medium|high|xhigh|max>",
        accepts_args=True,
    ),
    "/thinking": CommandEntry(
        handler=cmd_thinking,
        desc="Show current thinking state (on/off).",
        args_desc="on|off",
        accepts_args=True,
    ),
    "/plan": CommandEntry(
        handler=cmd_plan,
        desc="Enter Plan Mode: read-only inspection tools, no edits or shell.",
    ),
    "/review": CommandEntry(
        handler=cmd_review,
        desc="Enter Review Mode: read-only review, guarded validation.",
    ),
    "/act": CommandEntry(
        handler=cmd_act,
        desc="Enter Act Mode and allow normal tool use within policy.",
        accepts_args=True,
    ),
    "/verbose": CommandEntry(
        handler=cmd_verbose,
        desc="Show or hide tool call ids and result details.",
        args_desc="on|off",
        accepts_args=True,
    ),
    "/queue": CommandEntry(
        handler=cmd_queue,
        desc="Enable queued input while an agent turn is streaming.",
        args_desc="on|off",
        accepts_args=True,
    ),
    "/compact": CommandEntry(
        handler=cmd_compact,
        desc="Manually request context compaction and shrink the session log.",
    ),
    "/permissions": CommandEntry(
        handler=cmd_permissions,
        desc="List / revoke / clear permission rules.",
        accepts_args=True,
    ),
    "/tool": CommandEntry(
        handler=cmd_tool,
        desc="Run one registered tool directly, or list tools.",
        args_desc="NAME INPUT|list",
        accepts_args=True,
    ),
    "/exit": CommandEntry(handler=cmd_exit, desc="Exit the REPL."),
    "/quit": CommandEntry(handler=cmd_exit, desc="Exit the REPL.", visible=False),
}

COMMAND_NAMES = command_names(COMMAND_REGISTRY)
HELP_TEXT = generate_help_text(COMMAND_REGISTRY)


def handle_command(
    command: str,
    store: SessionStore,
    app: object,
    renderer,
    state,
    prompt_session,
    session_policy: SessionPermissionPolicy | None = None,
    persistent_store: PersistentPermissionStore | None = None,
) -> bool:
    ctx = CommandContext(
        store=store,
        app=app,
        renderer=renderer,
        state=state,
        prompt_session=prompt_session,
        session_policy=session_policy,
        persistent_store=persistent_store,
    )
    for prefix in sorted(COMMAND_REGISTRY, key=len, reverse=True):
        entry = COMMAND_REGISTRY[prefix]
        if command == prefix or (
            entry.accepts_args and command.startswith(prefix + " ")
        ):
            return entry.handler(command, ctx)
    print(f"Unknown command: {command}")
    return False
