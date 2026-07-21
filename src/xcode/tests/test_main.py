from types import SimpleNamespace

from xcode.main import _run, parse_args


def test_parse_args_supports_tui_and_cli_commands() -> None:
    assert parse_args(["tui"]).command == "tui"
    assert parse_args(["cli"]).command == "cli"


def test_default_command_is_tui(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("xcode.main._build_app_from_config", lambda *_: object())
    monkeypatch.setattr(
        "xcode.main.run_tui", lambda *args, **kwargs: calls.append("tui") or 0
    )
    monkeypatch.setattr(
        "xcode.main.run_repl", lambda *args, **kwargs: calls.append("cli") or 0
    )

    args = parse_args([])
    runtime_config = SimpleNamespace(paths=SimpleNamespace(sessions_dir=None))

    assert _run(args, runtime_config) == 0
    assert calls == ["tui"]


def test_cli_command_starts_repl(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("xcode.main._build_app_from_config", lambda *_: object())
    monkeypatch.setattr(
        "xcode.main.run_tui", lambda *args, **kwargs: calls.append("tui") or 0
    )
    monkeypatch.setattr(
        "xcode.main.run_repl", lambda *args, **kwargs: calls.append("cli") or 0
    )

    args = parse_args(["cli"])
    runtime_config = SimpleNamespace(paths=SimpleNamespace(sessions_dir=None))

    assert _run(args, runtime_config) == 0
    assert calls == ["cli"]
