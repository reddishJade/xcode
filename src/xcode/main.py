from __future__ import annotations

import argparse
from pathlib import Path

import sys

from .cli.repl import run_repl
from .cli.setup_wizard import has_valid_config, run_setup_wizard
from .harness.config import discover_runtime_config, resolve_config_path
from .harness.app import build_app


def parse_args() -> argparse.Namespace:
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
        "--setup", action="store_true", help="Force-run the provider setup wizard."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root
    temp_config: Path | None = None

    if args.setup or not has_valid_config(project_root):
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
        return run_repl(
            app, sessions_dir, resume_latest=args.resume, project_root=args.project_root
        )
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
