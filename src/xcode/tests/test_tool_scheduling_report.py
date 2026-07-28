"""工具调度 benchmark 配对报告测试。"""

from __future__ import annotations

from pathlib import Path

from benchmarks.reports.generate_tool_scheduling_report import (
    write_tool_scheduling_report,
)


def _record(
    variant: str,
    repeat: int,
    duration: float,
    *,
    digest: str = "same",
    success: bool = True,
) -> dict[str, object]:
    return {
        "task_id": "reads-05",
        "variant": variant,
        "repeat": repeat,
        "duration_seconds": duration,
        "call_count": 5,
        "read_calls": 5,
        "write_calls": 0,
        "tool_workers": 4,
        "controlled_delay_ms_total": 500,
        "max_concurrency": 1 if variant == "serial" else 4,
        "unsafe_overlap_events": 0,
        "output_digest": digest,
        "workspace_digest": digest,
        "success": success,
    }


def test_report_computes_per_task_p50_p95_and_paired_speedup(
    tmp_path: Path,
) -> None:
    records = [
        _record("serial", 1, 1.0),
        _record("xcode", 1, 0.25),
        _record("xcode", 2, 0.30),
        _record("serial", 2, 1.2),
    ]

    summary = write_tool_scheduling_report(records, tmp_path)

    assert summary["valid_pairs"] == 2
    tasks = summary["tasks"]
    assert isinstance(tasks, dict)
    stats = tasks["reads-05"]
    assert isinstance(stats, dict)
    assert stats["serial_p50_seconds"] == 1.1
    assert stats["xcode_p50_seconds"] == 0.275
    assert stats["p50_latency_reduction"] == 0.75
    assert stats["median_paired_speedup"] == 4.0
    quality = summary["quality"]
    assert isinstance(quality, dict)
    assert quality["serial_run_success_rate"] == 1.0
    assert quality["xcode_run_success_rate"] == 1.0
    assert quality["output_equivalence_rate"] == 1.0
    assert quality["workspace_equivalence_rate"] == 1.0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "n=2 pairs" in report
    assert "4.00x" in report


def test_report_excludes_failed_or_output_mismatched_pairs(tmp_path: Path) -> None:
    records = [
        _record("serial", 1, 1.0),
        _record("xcode", 1, 0.25, digest="different"),
        _record("serial", 2, 1.1),
        _record("xcode", 2, 0.3, success=False),
    ]

    summary = write_tool_scheduling_report(records, tmp_path)

    assert summary["valid_pairs"] == 0
    excluded = summary["excluded_pairs"]
    assert isinstance(excluded, list)
    assert len(excluded) == 2
    quality = summary["quality"]
    assert isinstance(quality, dict)
    assert quality["xcode_run_success_rate"] == 0.5
    assert quality["output_equivalence_rate"] == 0.5
    assert quality["workspace_equivalence_rate"] == 0.5
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "output digest mismatch" in report
    assert "xcode failed" in report
