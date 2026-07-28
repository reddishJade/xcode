"""长程上下文 benchmark 的离线测试。"""

from __future__ import annotations

from collections.abc import Iterator
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks.runners import _cli as benchmark_cli
from benchmarks.evaluators.state_retention import (
    capture_initial_state,
    evaluate_state_retention,
    retention_rate,
)
from benchmarks.models import CommandSpec, StateCheckSpec, load_task
from benchmarks.reports.generate_report import render_markdown, summarize_records
from benchmarks.runners._cli import _retryable_attempt
from benchmarks.runners._long_horizon import (
    _benchmark_runtime_config,
    _build_phase_metrics,
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


def test_report_uses_same_complete_pair_cohort_for_values_and_change() -> None:
    records = [
        _record("baseline", 1000, True, True, repeat=1),
        _record("xcode", 500, True, True, repeat=1),
        _record("baseline", 2000, True, True, repeat=2),
        {
            **_record("xcode", 400, True, False, repeat=2),
            "usage_incomplete_calls": [
                {
                    "call_index": 2,
                    "kind": "agent",
                    "error": "Request timed out",
                    "retryable": True,
                }
            ],
        },
    ]

    summary = summarize_records(records)
    markdown = render_markdown(summary)

    assert summary["cohorts"]["correctness_pairs"] == 2
    assert summary["cohorts"]["complete_usage_pairs"] == 1
    assert summary["variants"]["baseline"]["input_tokens_mean"] == 1000
    assert summary["variants"]["xcode"]["input_tokens_mean"] == 500
    assert summary["variants"]["baseline"]["usage_complete_runs"] == 2
    assert summary["variants"]["baseline"]["token_cohort_runs"] == 1
    assert summary["paired_changes"]["input_token_reduction"] == 0.5
    assert len(summary["excluded_attempts"]) == 1
    assert "n=1 pairs" in markdown
    assert "Excluded attempts" in markdown


def test_report_selects_first_complete_pair_attempt() -> None:
    records = [
        _record("baseline", 1000, True, True, attempt=1),
        {
            **_record("xcode", 200, True, False, attempt=1),
            "usage_incomplete_calls": [
                {
                    "call_index": 1,
                    "kind": "agent",
                    "error": "Request timed out",
                    "retryable": True,
                }
            ],
        },
        _record("baseline", 900, True, True, attempt=2),
        _record("xcode", 450, True, True, attempt=2),
    ]

    summary = summarize_records(records)

    assert summary["runs"] == 4
    assert summary["pair_attempts"] == 2
    assert summary["paired_runs"] == 1
    assert summary["retried_pairs"] == 1
    assert summary["selected_pairs"][0]["attempt"] == 2
    assert summary["variants"]["baseline"]["input_tokens_mean"] == 900
    assert summary["variants"]["xcode"]["input_tokens_mean"] == 450


def test_report_keeps_post_compaction_pair_when_total_usage_is_incomplete() -> None:
    baseline = {
        **_record("baseline", 1000, True, True),
        "post_compaction_input_tokens": 600,
        "post_compaction_usage_complete": True,
    }
    xcode = {
        **_record("xcode", 0, True, False),
        "post_compaction_input_tokens": 300,
        "post_compaction_usage_complete": True,
    }

    summary = summarize_records([baseline, xcode])

    assert summary["cohorts"]["complete_usage_pairs"] == 0
    assert summary["cohorts"]["post_compaction_usage_pairs"] == 1
    assert summary["paired_changes"]["input_token_reduction"] is None
    assert summary["paired_changes"]["post_compaction_input_token_reduction"] == 0.5


def test_phase_metrics_have_independent_usage_completeness() -> None:
    root = Path(__file__).resolve().parents[3]
    task = load_task(
        root / "benchmarks" / "tasks" / "long_horizon" / "parser_recovery" / "task.json"
    )
    turns: list[dict[str, object]] = []
    for turn in range(1, 11):
        calls = [_provider_call(10, has_usage=turn != 2)]
        if turn == 7:
            calls.insert(0, _provider_call(5, kind="compaction_summary"))
        turns.append({"turn": turn, "provider_calls": calls})

    metrics = _build_phase_metrics(task, turns)

    assert metrics["compaction_turn"] == 7
    assert metrics["restart_after_turn"] == 7
    assert metrics["pre_compaction_input_tokens"] == 60
    assert metrics["pre_compaction_usage_complete"] is False
    assert metrics["post_compaction_input_tokens"] == 45
    assert metrics["post_compaction_usage_complete"] is True
    assert metrics["post_resume_input_tokens"] == 30
    assert metrics["post_resume_usage_complete"] is True
    assert metrics["compaction_summary_input_tokens"] == 5


def test_transient_incomplete_usage_retries_but_missing_usage_does_not() -> None:
    complete = _record("baseline", 1000, True, True)
    transient = {
        **_record("xcode", 500, True, False),
        "retryable_usage_failure": True,
    }
    unexplained = {
        **_record("xcode", 500, True, False),
        "retryable_usage_failure": False,
    }

    assert _retryable_attempt([complete, transient]) is True
    assert _retryable_attempt([complete, unexplained]) is False


def test_cli_retries_the_whole_pair_and_preserves_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = SimpleNamespace(id="task-a")
    reporter = _Reporter()

    monkeypatch.setattr(benchmark_cli, "_runtime_config", lambda _path: object())
    monkeypatch.setattr(
        benchmark_cli,
        "discover_task_files",
        lambda _paths: (tmp_path / "task.json",),
    )
    monkeypatch.setattr(benchmark_cli, "load_task", lambda _path: task)
    monkeypatch.setattr(
        benchmark_cli,
        "create_progress_reporter",
        lambda _total, enabled: reporter,
    )

    def fake_run_task(
        _task: object,
        variant: str,
        _runtime_config: object,
        options: object,
    ) -> dict[str, object]:
        attempt = int(getattr(options, "attempt"))
        incomplete = variant == "xcode" and attempt == 1
        return {
            **_record(variant, 100, True, not incomplete, attempt=attempt),
            "retryable_usage_failure": incomplete,
            "usage_incomplete_calls": (
                [{"error": "Request timed out"}] if incomplete else []
            ),
        }

    monkeypatch.setattr(benchmark_cli, "run_task", fake_run_task)
    args = argparse.Namespace(
        repeat=1,
        max_pair_attempts=2,
        config=None,
        tasks=[tmp_path],
        no_progress=False,
        output_dir=tmp_path / "results",
        temperature=0,
        summary_mode="model",
        keep_workspaces=False,
    )

    records = benchmark_cli._run_variants(args, ("baseline", "xcode"))

    assert [(record["variant"], record["attempt"]) for record in records] == [
        ("baseline", 1),
        ("xcode", 1),
        ("baseline", 2),
        ("xcode", 2),
    ]
    assert reporter.added_runs == 2


def test_strict_report_exits_after_writing_incomplete_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark_cli,
        "write_report",
        lambda _records, _output: {
            "paired_runs": 2,
            "cohorts": {"complete_usage_pairs": 1},
        },
    )
    args = argparse.Namespace(
        output_dir=tmp_path,
        require_complete_usage=True,
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_cli._write_and_validate_report(
            [],
            args,
            ("baseline", "xcode"),
        )

    assert raised.value.code == 2


def test_report_rejects_mismatched_pair_controls() -> None:
    baseline = {**_record("baseline", 1000, True, True), "model": "model-a"}
    xcode = {**_record("xcode", 500, True, True), "model": "model-b"}

    with pytest.raises(ValueError, match="differ on control model"):
        summarize_records([baseline, xcode])


def test_report_keeps_usage_for_single_variant_runner() -> None:
    summary = summarize_records([_record("baseline", 1000, True, True)])

    assert summary["paired_runs"] == 0
    assert summary["variants"]["baseline"]["runs"] == 1
    assert summary["variants"]["baseline"]["input_tokens_mean"] == 1000


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
    *,
    repeat: int = 1,
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "task_id": "task-a",
        "variant": variant,
        "repeat": repeat,
        "attempt": attempt,
        "model": "test-model",
        "temperature": 0,
        "execution_mode": "build",
        "summary_mode": "model",
        "baseline_commit": "fixture-commit",
        "usage_complete": usage_complete,
        "usage_incomplete_calls": [],
        "retryable_usage_failure": False,
        "input_tokens_total": tokens,
        "peak_input_tokens": tokens,
        "input_cost_usd": tokens / 1_000_000,
        "task_success": completed,
        "long_session_completed": completed,
        "state_retention": 1.0,
        "context_overflow": False,
        "duration_seconds": 1.0,
    }


def _provider_call(
    input_tokens: int,
    *,
    has_usage: bool = True,
    kind: str = "agent",
) -> dict[str, object]:
    return {
        "kind": kind,
        "model": "test-model",
        "input_tokens": input_tokens,
        "output_tokens": 1,
        "duration_seconds": 0.1,
        "has_usage": has_usage,
        "error": None,
    }


class _Reporter:
    def __init__(self) -> None:
        self.added_runs = 0

    def __enter__(self) -> _Reporter:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, _event: object) -> None:
        return None

    def add_runs(self, count: int) -> None:
        self.added_runs += count


def _git(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()
