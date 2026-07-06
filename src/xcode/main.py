from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .cli.config_cmd import handle_config_command
from .cli.memory_cmd import handle_memory_command
from .cli.repl import run_repl
from .cli.setup_wizard import has_valid_config, run_setup_wizard
from .harness.app import build_app
from .harness.config import discover_runtime_config, resolve_config_path


def _build_config_parser(subparsers) -> None:
    config_parser = subparsers.add_parser(
        "config",
        help="Manage provider API profiles (list / add / edit / delete / set)",
        description=(
            "Manage provider API profiles. Each profile stores transport, model, "
            "API key and other settings for an LLM provider. "
            "Profiles are stored in xcode.config.json.\n\n"
            "Available subcommands:\n"
            "  list    — show all configured profiles\n"
            "  add     — create a new profile interactively\n"
            "  edit    — modify an existing profile interactively\n"
            "  delete  — remove a profile\n"
            "  set     — quick field update without interactive prompts\n\n"
            "Common profile fields (set via `config set <profile> <field> <value>`):\n"
            "  transport         Provider type: openai_chat, deepseek_chat, "
            "chatglm_chat, mimo_chat\n"
            "  chat_model        Model ID, e.g. deepseek-v4-flash, gpt-5.5\n"
            "  base_url          API base URL\n"
            "  api_key           API key (or set via env var instead)\n"
            "  thinking          Enable reasoning mode (true/false)\n"
            "  reasoning_effort  Level: off/minimal/low/medium/high/xhigh/max\n"
            "  clear_thinking    Strip thinking tags from output (true/false)\n"
            "  tool_stream       Stream tool calls incrementally (true/false)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory."
    )
    config_parser.add_argument(
        "--config", type=Path, help="Path to xcode.config.json to manage."
    )
    config_sub = config_parser.add_subparsers(dest="config_action")

    list_p = config_sub.add_parser(
        "list", help="Show all configured profiles and their settings"
    )
    list_p.set_defaults(config_action="list")

    add_p = config_sub.add_parser(
        "add", help="Create a new provider profile via interactive prompts"
    )
    add_p.add_argument("name", help="Profile name, e.g. main, subagent, fallback")
    add_p.set_defaults(config_action="add")

    edit_p = config_sub.add_parser(
        "edit", help="Modify an existing provider profile via interactive prompts"
    )
    edit_p.add_argument("name", help="Profile name to edit")
    edit_p.set_defaults(config_action="edit")

    delete_p = config_sub.add_parser("delete", help="Remove a provider profile by name")
    delete_p.add_argument("name", help="Profile name to delete")
    delete_p.set_defaults(config_action="delete")

    set_p = config_sub.add_parser(
        "set",
        help="Set a single field in a profile (non-interactive, for scripting)",
        description=(
            "Set a single field value in a profile without interactive prompts. "
            "Useful for quick changes or scripting.\n\n"
            "Available fields: transport, chat_model, base_url, api_key, "
            "thinking (bool), reasoning_effort, clear_thinking, tool_stream"
        ),
    )
    set_p.add_argument("name", help="Profile name")
    set_p.add_argument("field", help="Field name (see above)")
    set_p.add_argument("value", help="Field value")
    set_p.set_defaults(config_action="set")


def _build_setup_parser(subparsers) -> None:
    subparsers.add_parser("setup", help="Run the provider setup wizard")


def _build_memory_parser(subparsers) -> None:
    memory_parser = subparsers.add_parser(
        "memory",
        help="Inspect and explicitly manage durable-memory proposals",
        description=(
            "Inspect governed durable memory without starting an agent. "
            "Approval, rejection, and retirement are direct user actions and "
            "require --yes.\n\n"
            "Use --project-root before the memory command, for example:\n"
            "  xcode --project-root . memory proposals\n"
            "  xcode --project-root . memory approve mp_... --yes\n"
            "  xcode --project-root . memory retire mem_... --yes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    memory_parser.set_defaults(memory_action="proposals", status="pending")
    memory_sub = memory_parser.add_subparsers(dest="memory_action")

    proposals_p = memory_sub.add_parser(
        "proposals", help="List project and user memory proposals"
    )
    proposals_p.add_argument(
        "--status",
        choices=["all", "pending", "approved", "rejected", "applied", "failed"],
        default="pending",
        help="Proposal status filter.",
    )

    approve_p = memory_sub.add_parser(
        "approve", help="Approve one pending proposal and write its memory"
    )
    approve_p.add_argument("proposal_id", help="The mp_* proposal identifier.")
    approve_p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm this durable memory write.",
    )
    approve_p.add_argument(
        "--approver",
        default="cli_user",
        help="Audit label for the user who approves the proposal.",
    )

    reject_p = memory_sub.add_parser(
        "reject", help="Reject one pending proposal without writing memory"
    )
    reject_p.add_argument("proposal_id", help="The mp_* proposal identifier.")
    reject_p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm this rejection.",
    )
    reject_p.add_argument(
        "--reason",
        default="rejected_by_user",
        help="Reason recorded in the proposal ledger.",
    )

    retire_p = memory_sub.add_parser(
        "retire", help="Archive one active memory through a retirement proposal"
    )
    retire_p.add_argument("memory_id", help="The mem_* identifier to retire.")
    retire_p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm this retirement and archive operation.",
    )
    retire_p.add_argument(
        "--reason",
        default="retired_by_user",
        help="Reason written to the retirement archive and ledger evidence.",
    )

    memory_sub.add_parser(
        "audit",
        help="Check durable memory records against proposal and evidence provenance",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xcode coding agent.")
    parser.add_argument(
        "-p", "--prompt", help="Run one prompt and exit (single-shot mode)."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory."
    )
    parser.add_argument(
        "--config", type=Path, help="Path to xcode.config.json runtime settings."
    )
    parser.add_argument(
        "--sessions-dir", type=Path, help="REPL session transcript directory."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Open the REPL resume picker on startup."
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_",
        help="Resume the latest session for the current project.",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Resume a specific session by id.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_config_parser(subparsers)
    _build_setup_parser(subparsers)
    _build_memory_parser(subparsers)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    project_root = args.project_root

    if args.command == "config":
        handle_config_command(args, project_root)
        return 0

    if args.command == "setup":
        try:
            run_setup_wizard(project_root)
        except KeyboardInterrupt:
            pass
        return 0

    if args.command == "memory":
        return handle_memory_command(args, project_root)

    temp_config: Path | None = None
    if not has_valid_config(project_root):
        if sys.stdin.isatty():
            try:
                status, config_path = run_setup_wizard(project_root)
            except KeyboardInterrupt:
                return 0
            if status == "cancelled":
                return 0
            if status == "no_save" and config_path is not None:
                temp_config = config_path
                args.config = config_path
        elif not has_valid_config(project_root):
            print(
                "No API key configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or DEEPSEEK_API_KEY in .env or environment.",
                file=sys.stderr,
            )

    try:
        runtime_config = discover_runtime_config(project_root, args.config)
        return _run(args, runtime_config)
    finally:
        if temp_config is not None and temp_config.exists():
            temp_config.unlink()


def _run(args, runtime_config) -> int:
    app = _build_app_from_config(args.project_root, runtime_config)
    if args.prompt:
        _print_stream(app.ask_stream(args.prompt))
        return 0
    sessions_dir = (
        args.sessions_dir
        or resolve_config_path(args.project_root, runtime_config.paths.sessions_dir)
        or (args.project_root / ".local" / "sessions")
    )
    daemon = getattr(app, "daemon", None)
    if daemon is not None:
        daemon.start()
    try:
        if args.session:
            return run_repl(
                app,
                sessions_dir,
                session_id=args.session,
                project_root=args.project_root,
            )
        if args.continue_:
            return run_repl(
                app,
                sessions_dir,
                auto_continue=True,
                project_root=args.project_root,
            )
        if args.resume:
            return run_repl(
                app,
                sessions_dir,
                resume_latest=True,
                project_root=args.project_root,
            )
        return run_repl(app, sessions_dir, project_root=args.project_root)
    finally:
        if daemon is not None:
            daemon.stop()


def _build_app_from_config(project_root: Path, runtime_config):
    return build_app(
        project_root=project_root,
        runtime_config=runtime_config,
    )


def _print_stream(events) -> None:
    answer_parts = []
    for event in events:
        if event.type == "text_delta":
            print(str(event.data), end="", flush=True)
            answer_parts.append(str(event.data))
        elif event.type == "final" and not answer_parts:
            print(event.data.answer)
            answer_parts.append(event.data.answer)
    if answer_parts:
        print()


if __name__ == "__main__":
    sys.exit(main())
