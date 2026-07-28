"""长程 benchmark 进度展示测试。"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from benchmarks.runners.progress import (
    NullBenchmarkProgress,
    PlainBenchmarkProgress,
    ProgressStage,
    ProgressUpdate,
    RichBenchmarkProgress,
    create_progress_reporter,
)


def _event(
    stage: ProgressStage,
    *,
    turn: int | None = None,
    detail: str = "",
) -> ProgressUpdate:
    return ProgressUpdate(
        stage=stage,
        task_id="parser-recovery",
        variant="baseline",
        repeat=1,
        total_turns=10,
        turn=turn,
        detail=detail,
    )


def test_plain_progress_flushes_run_turn_and_tool_status() -> None:
    output = StringIO()
    reporter = PlainBenchmarkProgress(total_runs=2, stream=output)

    with reporter:
        reporter.update(_event("run_started"))
        reporter.update(_event("turn_started", turn=1, detail="inspect code"))
        reporter.update(_event("tool_started", turn=1, detail="read_file · src/a.py"))
        reporter.update(_event("run_completed", detail="success=True"))

    rendered = output.getvalue()
    assert "run 0/2 parser-recovery baseline r1" in rendered
    assert "turn 1/10" in rendered
    assert "read_file · src/a.py" in rendered
    assert "run 1/2" in rendered


def test_factory_uses_plain_progress_for_non_terminal_console() -> None:
    console = Console(file=StringIO(), force_terminal=False)

    reporter = create_progress_reporter(1, console=console)

    assert isinstance(reporter, PlainBenchmarkProgress)


def test_factory_can_disable_progress() -> None:
    reporter = create_progress_reporter(1, enabled=False)

    assert isinstance(reporter, NullBenchmarkProgress)


def test_rich_progress_accepts_live_updates() -> None:
    console = Console(file=StringIO(), force_terminal=True, width=140)
    reporter = RichBenchmarkProgress(total_runs=1, console=console)

    with reporter:
        reporter.update(_event("run_started"))
        reporter.update(_event("turn_started", turn=1))
        reporter.update(_event("provider_started", turn=1, detail="agent request"))
        reporter.update(_event("turn_completed", turn=1))
        reporter.update(_event("run_completed", detail="success=True"))

    assert "parser-recovery baseline r1" in str(console.file.getvalue())
