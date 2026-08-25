from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli.config_cmd import handle_config_command
from .cli.repl import run_repl
from .cli.setup_wizard import has_valid_config, run_setup_wizard
from .cli.tui import run_tui
from .coding_agent.app import build_app
from .harness.config import discover_runtime_config, resolve_config_path


def _build_config_parser(subparsers) -> None:
    config_parser = subparsers.add_parser(
        "config",
        help="Browse and edit xcode settings interactively",
        description=(
            "Open an interactive browser over all xcode.config.json settings: "
            "execution modes, agent limits, request hygiene, security policy, "
            "paths, tools and skills. Pick a row to cycle preset values or "
            "type a new one; changes are validated before they are saved."
        ),
    )
    config_parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory."
    )
    config_parser.add_argument(
        "--config", type=Path, help="Path to xcode.config.json to manage."
    )


def _build_setup_parser(subparsers) -> None:
    subparsers.add_parser("setup", help="Run the provider setup wizard")


def _build_tui_parser(subparsers) -> None:
    subparsers.add_parser("tui", help="Run the full-screen terminal UI")


def _build_web_parser(subparsers) -> None:
    web_parser = subparsers.add_parser(
        "web",
        help="Run the browser workbench (FastAPI + WebSocket UI)",
        description=(
            "Start the Xcode browser workbench: a single-page frontend that "
            "streams agent events over WebSocket."
        ),
    )
    web_parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host address (default: 127.0.0.1)."
    )
    web_parser.add_argument(
        "--port", type=int, default=8787, help="Bind port (default: 8787)."
    )
    web_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the browser automatically after the server starts.",
    )
    web_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory.",
    )


def _build_cli_parser(subparsers) -> None:
    subparsers.add_parser("cli", help="Run the interactive command-line REPL")


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
        "--sessions-dir", type=Path, help="Session transcript directory."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Open the session resume picker on startup.",
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
    _build_tui_parser(subparsers)
    _build_cli_parser(subparsers)
    _build_web_parser(subparsers)
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
        else:
            if not has_valid_config(project_root):
                print(
                    "No API key configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                    "or DEEPSEEK_API_KEY in .env or environment.",
                    file=sys.stderr,
                )

    try:
        runtime_config = discover_runtime_config(project_root, args.config)
        if args.command == "web":
            from .server.serve import run_web_server

            return run_web_server(
                project_root,
                host=args.host,
                port=args.port,
                config_path=args.config,
                runtime_config=runtime_config,
                open_browser=args.open,
            )
        return _run(args, runtime_config)
    finally:
        if temp_config is not None and temp_config.exists():
            temp_config.unlink()


def _run(args, runtime_config) -> int:
    sessions_dir = (
        args.sessions_dir
        or resolve_config_path(args.project_root, runtime_config.paths.sessions_dir)
        or (args.project_root / ".xcode" / "sessions")
    )
    app = _build_app_from_config(args.project_root, runtime_config, sessions_dir)
    if args.prompt:
        _print_stream(app.ask_stream(args.prompt))
        return 0
    if args.command == "tui":
        return run_tui(
            app,
            args.project_root,
            session_id=args.session,
            auto_continue=args.continue_,
            resume_latest=args.resume,
        )
    if args.command == "cli":
        if args.session:
            return run_repl(
                app,
                session_id=args.session,
                project_root=args.project_root,
            )
        if args.continue_:
            return run_repl(app, auto_continue=True, project_root=args.project_root)
        if args.resume:
            return run_repl(app, resume_latest=True, project_root=args.project_root)
        return run_repl(app, project_root=args.project_root)

    return run_tui(
        app,
        args.project_root,
        session_id=args.session,
        auto_continue=args.continue_,
        resume_latest=args.resume,
    )


def _build_app_from_config(
    project_root: Path,
    runtime_config,
    sessions_dir: Path,
):
    return build_app(
        project_root=project_root,
        runtime_config=runtime_config,
        sessions_dir=sessions_dir,
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
