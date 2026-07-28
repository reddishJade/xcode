"""长程 benchmark 进度展示测试。"""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, cast

import pytest
from rich.console import Console

from benchmarks.runners import _long_horizon
from benchmarks.runners._long_horizon import InstrumentedProvider
from benchmarks.runners.progress import (
    NullBenchmarkProgress,
    PlainBenchmarkProgress,
    ProgressStage,
    ProgressUpdate,
    RichBenchmarkProgress,
    create_progress_reporter,
)
from xcode.ai.events import FinalMessage, ReasoningDelta, TextDelta, UsageUpdate


def _event(
    stage: ProgressStage,
    *,
    turn: int | None = None,
    detail: str = "",
    attempt: int = 1,
) -> ProgressUpdate:
    return ProgressUpdate(
        stage=stage,
        task_id="parser-recovery",
        variant="baseline",
        repeat=1,
        attempt=attempt,
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


def test_plain_progress_expands_total_and_labels_retry_attempt() -> None:
    output = StringIO()
    reporter = PlainBenchmarkProgress(total_runs=2, stream=output)

    with reporter:
        reporter.add_runs(2)
        reporter.update(_event("run_started", attempt=2))

    rendered = output.getvalue()
    assert "run 0/4" in rendered
    assert "parser-recovery baseline r1 a2" in rendered


def test_plain_progress_throttles_provider_heartbeats() -> None:
    output = StringIO()
    reporter = PlainBenchmarkProgress(total_runs=1, stream=output)

    with reporter:
        reporter.update(_event("provider_started", turn=1, detail="agent #1"))
        reporter.update(
            _event("provider_streaming", turn=1, detail="agent #1 · reasoning")
        )
        reporter.update(
            _event("provider_streaming", turn=1, detail="agent #1 · reasoning")
        )

    rendered = output.getvalue()
    assert rendered.count("model request active") == 1


def test_rich_progress_accepts_live_updates() -> None:
    console = Console(file=StringIO(), force_terminal=True, width=140)
    reporter = RichBenchmarkProgress(total_runs=1, console=console)

    with reporter:
        reporter.update(_event("run_started"))
        reporter.update(_event("turn_started", turn=1))
        reporter.update(_event("provider_started", turn=1, detail="agent request"))
        reporter.update(
            _event("provider_streaming", turn=1, detail="agent #1 · reasoning")
        )
        reporter.update(_event("turn_completed", turn=1))
        reporter.update(_event("run_completed", detail="success=True"))

    assert "parser-recovery baseline r1" in str(console.file.getvalue())


async def test_instrumented_provider_reports_stream_activity_and_call_number() -> None:
    updates: list[tuple[ProgressStage, str]] = []
    calls = []
    provider = InstrumentedProvider(
        cast(Any, _StreamingProvider()),
        0,
        calls,
        lambda stage, detail: updates.append((stage, detail)),
    )

    events = [event async for event in provider.stream([], [])]

    assert len(events) == 4
    assert calls[0].input_tokens == 100
    assert calls[0].output_tokens == 20
    assert updates[0][0] == "provider_started"
    assert "summary #1" in updates[0][1]
    streaming = [detail for stage, detail in updates if stage == "provider_streaming"]
    assert any("reasoning" in detail for detail in streaming)
    assert any("answer" in detail for detail in streaming)
    assert updates[-1][0] == "provider_finished"
    assert "summary #1" in updates[-1][1]


async def test_instrumented_provider_heartbeats_before_first_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[tuple[ProgressStage, str]] = []
    monkeypatch.setattr(_long_horizon, "_PROVIDER_HEARTBEAT_SECONDS", 0.005)
    provider = InstrumentedProvider(
        cast(Any, _DelayedStreamingProvider()),
        0,
        [],
        lambda stage, detail: updates.append((stage, detail)),
    )

    _ = [event async for event in provider.stream([], [])]

    streaming = [detail for stage, detail in updates if stage == "provider_streaming"]
    assert any("waiting first event" in detail for detail in streaming)
    assert any("no events yet" in detail for detail in streaming)


class _StreamingProvider:
    model = "test-model"
    base_url = "https://example.invalid"
    transport = "test"
    thinking = True
    reasoning_effort = "high"

    async def stream(self, *args: object, **kwargs: object):
        yield ReasoningDelta("thinking")
        yield TextDelta("answer")
        yield UsageUpdate(input_tokens=100, output_tokens=20)
        yield FinalMessage(content="answer", stop_reason="end_turn")


class _DelayedStreamingProvider(_StreamingProvider):
    async def stream(self, *args: object, **kwargs: object):
        await asyncio.sleep(0.02)
        yield UsageUpdate(input_tokens=100, output_tokens=20)
        yield FinalMessage(content="answer", stop_reason="end_turn")
