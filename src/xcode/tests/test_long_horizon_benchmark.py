"""长程上下文 benchmark 的离线测试。"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks.evaluators.state_retention import (
    capture_initial_state,
    evaluate_state_retention,
    retention_rate,
)
from benchmarks.models import CommandSpec, StateCheckSpec, load_task
from benchmarks.reports.generate_report import render_markdown, summarize_records
from benchmarks.runners._long_horizon import (
    _benchmark_runtime_config,
    _prepare_workspace,
    _repeated_read_calls,
    _run_turn,
)
from xcode.harness.config import XcodeRuntimeConfig
from xcode.harness.agent_runtime.events import FinalStructuredEvent
from xcode.harness.agent_runtime.result import AgentHarnessResult


def test_example_task_has_compaction_restart_and_ten_turns() -> None:
    root = Path(__file__).resolve().parents[3]
    task = load_task(
        root / "benchmarks" / "tasks" / "long_horizon" / "parser_recovery" / "task.json"
    )

    assert len(task.turns) == 10
    assert any(turn.compact_before for turn in task.turns)
    assert any(turn.restart_after for turn in task.turns)
    assert task.success_command.argv[:3] == ("python", "-m", "unittest")


def test_state_retention_uses_hashes_and_commands(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    protected = tmp_path / "protected.py"
    source.write_text("before\n", encoding="utf-8")
    protected.write_text("stable\n", encoding="utf-8")
    checks = (
        StateCheckSpec(id="changed", kind="file_changed", path="source.py"),
        StateCheckSpec(id="protected", kind="file_unchanged", path="protected.py"),
        StateCheckSpec(id="absent", kind="path_absent", path="forbidden.py"),
        StateCheckSpec(
            id="command",
            kind="command_succeeds",
            command=CommandSpec(argv=("python", "-c", "print('ok')")),
        ),
    )
    initial = capture_initial_state(tmp_path, checks)
    source.write_text("after\n", encoding="utf-8")

    outcomes = evaluate_state_retention(tmp_path, checks, initial)

    assert all(outcome.passed for outcome in outcomes)
    assert retention_rate(outcomes) == 1.0


def test_task_loader_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = {
        "schema_version": 1,
        "id": "invalid-task",
        "workspace": "workspace",
        "turns": [{"prompt": "inspect"}],
        "success_command": {"argv": ["python", "-V"]},
        "unexpected": True,
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        load_task(path)


def test_report_uses_only_paired_changes() -> None:
    records = [
        _record("baseline", 1000, False, True),
        _record("xcode", 400, True, True),
        {
            **_record("baseline", 9999, False, False),
            "task_id": "unpaired",
        },
    ]

    summary = summarize_records(records)
    markdown = render_markdown(summary)

    assert summary["paired_runs"] == 1
    assert summary["paired_changes"]["input_token_reduction"] == 0.6
    assert "60.0%" in markdown
    assert "Task success is determined" in markdown


def test_report_rejects_mismatched_pair_controls() -> None:
    baseline = {**_record("baseline", 1000, True, True), "model": "model-a"}
    xcode = {**_record("xcode", 500, True, True), "model": "model-b"}

    with pytest.raises(ValueError, match="differ on control model"):
        summarize_records([baseline, xcode])


def test_repeated_read_calls_count_same_path_only() -> None:
    calls = [
        {"name": "read_file", "input": {"path": "src/a.py"}},
        {"name": "read_file", "input": {"path": "src/a.py"}},
        {"name": "read_file", "input": {"path": "src/b.py"}},
        {"name": "edit_file", "input": {"path": "src/a.py"}},
    ]

    assert _repeated_read_calls(calls) == 1


def test_benchmark_turn_forces_non_interactive_build_mode() -> None:
    modes: list[str | None] = []
    result = AgentHarnessResult(answer="done", messages=[], steps=1, tool_calls=[])

    class _Agent:
        def run_stream(
            self, prompt: str, mode: str | None = None
        ) -> Iterator[FinalStructuredEvent]:
            assert prompt == "fix it"
            modes.append(mode)
            yield FinalStructuredEvent("final", 1, result)

    app = cast(Any, SimpleNamespace(agent=_Agent()))

    actual, compactions = _run_turn(app, "fix it")

    assert actual is result
    assert compactions == 0
    assert modes == ["build"]


def test_workspace_git_baseline_is_isolated_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "fixture"
    source.mkdir()
    (source / "source.py").write_text("before\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "must-not-copy").write_text("outer\n", encoding="utf-8")

    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init", "--quiet")
    first = parent / "results" / "run-1"
    second = parent / "results" / "run-2"
    first.parent.mkdir()

    first_commit = _prepare_workspace(source, first)
    second_commit = _prepare_workspace(source, second)

    assert first_commit == second_commit
    assert Path(_git(first, "rev-parse", "--show-toplevel")) == first
    assert not (first / ".git" / "must-not-copy").exists()
    assert _git(first, "status", "--short") == ""
    assert _git(first, "show", "HEAD:source.py") == "before"

    (first / "source.py").write_text("after\n", encoding="utf-8")
    session_dir = first / ".benchmark" / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")

    assert _git(first, "status", "--short") == "M source.py"
    diff = _git(first, "diff", "HEAD", "--", "source.py")
    assert "-before" in diff
    assert "+after" in diff


def test_benchmark_runtime_state_is_kept_outside_workspace(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    task = load_task(
        root / "benchmarks" / "tasks" / "long_horizon" / "parser_recovery" / "task.json"
    )
    sessions_dir = tmp_path / "runtime" / "sessions"

    configured = _benchmark_runtime_config(
        XcodeRuntimeConfig(),
        task,
        sessions_dir=sessions_dir,
    )

    assert configured.paths.sessions_dir == sessions_dir
    instructions = configured.prompt.instructions
    assert instructions[-1].content is not None
    assert "git diff HEAD" in instructions[-1].content
    assert "do not inspect benchmark transcripts" in instructions[-1].content


def _record(
    variant: str,
    tokens: int,
    completed: bool,
    usage_complete: bool,
) -> dict[str, object]:
    return {
        "task_id": "task-a",
        "variant": variant,
        "repeat": 1,
        "usage_complete": usage_complete,
        "input_tokens_total": tokens,
        "peak_input_tokens": tokens,
        "input_cost_usd": tokens / 1_000_000,
        "task_success": completed,
        "long_session_completed": completed,
        "state_retention": 1.0,
        "context_overflow": False,
        "duration_seconds": 1.0,
    }


def _git(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()
