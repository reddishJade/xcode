"""工具调度消融实验的配对统计与 Markdown 报告。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def write_tool_scheduling_report(
    records: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    """生成机器可读摘要和按 workload 分组的配对报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, excluded = _select_pairs(records)
    by_task: dict[str, list[_Pair]] = defaultdict(list)
    for pair in pairs:
        by_task[pair.task_id].append(pair)
    task_summaries = {
        task_id: _summarize_pairs(task_pairs)
        for task_id, task_pairs in sorted(by_task.items())
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "runs": len(records),
        "logical_pairs": len(
            {
                (str(record.get("task_id")), _integer(record, "repeat"))
                for record in records
            }
        ),
        "valid_pairs": len(pairs),
        "excluded_pairs": excluded,
        "overall": _summarize_pairs(pairs),
        "tasks": task_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return summary


def load_tool_scheduling_records(path: Path) -> list[dict[str, object]]:
    """读取结果目录中的原始运行记录，忽略 summary.json。"""
    records: list[dict[str, object]] = []
    for record_path in sorted(path.resolve().glob("*.json")):
        if record_path.name == "summary.json":
            continue
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "variant" in payload:
            records.append({str(key): value for key, value in payload.items()})
    if not records:
        raise ValueError(f"no tool scheduling records found in {path.resolve()}")
    return records


class _Pair:
    def __init__(
        self,
        task_id: str,
        repeat: int,
        serial: dict[str, object],
        xcode: dict[str, object],
    ) -> None:
        self.task_id = task_id
        self.repeat = repeat
        self.serial = serial
        self.xcode = xcode


def _select_pairs(
    records: list[dict[str, object]],
) -> tuple[list[_Pair], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        task_id = str(record.get("task_id", ""))
        repeat = _integer(record, "repeat")
        variant = str(record.get("variant", ""))
        grouped[(task_id, repeat)][variant].append(record)
    selected: list[_Pair] = []
    excluded: list[dict[str, object]] = []
    for (task_id, repeat), variants in sorted(grouped.items()):
        reasons: list[str] = []
        serial_records = variants.get("serial", [])
        xcode_records = variants.get("xcode", [])
        if len(serial_records) != 1:
            reasons.append(f"serial records={len(serial_records)}")
        if len(xcode_records) != 1:
            reasons.append(f"xcode records={len(xcode_records)}")
        if not reasons:
            serial = serial_records[0]
            xcode = xcode_records[0]
            if not bool(serial.get("success")):
                reasons.append("serial failed")
            if not bool(xcode.get("success")):
                reasons.append("xcode failed")
            if serial.get("output_digest") != xcode.get("output_digest"):
                reasons.append("output digest mismatch")
            if _integer(serial, "call_count") != _integer(xcode, "call_count"):
                reasons.append("call count mismatch")
            if not reasons:
                selected.append(_Pair(task_id, repeat, serial, xcode))
        if reasons:
            excluded.append(
                {
                    "task_id": task_id,
                    "repeat": repeat,
                    "reasons": reasons,
                }
            )
    return selected, excluded


def _summarize_pairs(pairs: list[_Pair]) -> dict[str, object]:
    if not pairs:
        return {
            "pairs": 0,
            "serial_p50_seconds": None,
            "xcode_p50_seconds": None,
            "p50_latency_reduction": None,
            "serial_p95_seconds": None,
            "xcode_p95_seconds": None,
            "p95_latency_reduction": None,
            "median_paired_speedup": None,
            "median_paired_latency_reduction": None,
            "xcode_max_concurrency": None,
            "write_isolation_rate": None,
            "output_equivalence_rate": None,
        }
    serial_durations = [_number(pair.serial, "duration_seconds") for pair in pairs]
    xcode_durations = [_number(pair.xcode, "duration_seconds") for pair in pairs]
    serial_p50 = _percentile(serial_durations, 0.5)
    xcode_p50 = _percentile(xcode_durations, 0.5)
    serial_p95 = _percentile(serial_durations, 0.95)
    xcode_p95 = _percentile(xcode_durations, 0.95)
    speedups = [
        serial / xcode
        for serial, xcode in zip(serial_durations, xcode_durations, strict=True)
        if xcode > 0
    ]
    reductions = [
        _reduction(serial, xcode)
        for serial, xcode in zip(serial_durations, xcode_durations, strict=True)
    ]
    return {
        "pairs": len(pairs),
        "call_count": _integer(pairs[0].serial, "call_count"),
        "tool_workers": _integer(pairs[0].xcode, "tool_workers"),
        "read_calls": _integer(pairs[0].xcode, "read_calls"),
        "write_calls": _integer(pairs[0].xcode, "write_calls"),
        "controlled_delay_ms_total": _number(
            pairs[0].xcode, "controlled_delay_ms_total"
        ),
        "serial_p50_seconds": serial_p50,
        "xcode_p50_seconds": xcode_p50,
        "p50_latency_reduction": _reduction(serial_p50, xcode_p50),
        "serial_p95_seconds": serial_p95,
        "xcode_p95_seconds": xcode_p95,
        "p95_latency_reduction": _reduction(serial_p95, xcode_p95),
        "median_paired_speedup": statistics.median(speedups),
        "median_paired_latency_reduction": statistics.median(reductions),
        "xcode_max_concurrency": max(
            _integer(pair.xcode, "max_concurrency") for pair in pairs
        ),
        "write_isolation_rate": statistics.mean(
            float(_integer(pair.xcode, "unsafe_overlap_events") == 0) for pair in pairs
        ),
        "output_equivalence_rate": statistics.mean(
            float(pair.serial.get("output_digest") == pair.xcode.get("output_digest"))
            for pair in pairs
        ),
    }


def _render_report(summary: dict[str, object]) -> str:
    lines = [
        "# Tool scheduling ablation",
        "",
        (
            f"Runs: {summary['runs']} records, {summary['logical_pairs']} logical "
            f"pairs, {summary['valid_pairs']} valid performance pairs."
        ),
        "",
        "This benchmark replays identical deterministic tool-call batches through ",
        "the production scheduler. Serial forces every call to run sequentially; ",
        "Xcode honors each tool's parallel/sequential execution classification.",
        "Controlled delay models reproducible I/O waiting and is not model latency.",
        "",
        (
            "| Workload | Calls (R/W) | Workers | Cohort | Serial P50 | Xcode P50 "
            "| P50 reduction | Serial P95 | Xcode P95 | P95 reduction | Median "
            "speedup | Xcode max concurrency |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    raw_tasks = summary.get("tasks")
    tasks = raw_tasks if isinstance(raw_tasks, dict) else {}
    for task_id, raw_stats in tasks.items():
        stats = raw_stats if isinstance(raw_stats, dict) else {}
        lines.append(
            f"| {task_id} | {stats.get('call_count', 0)} "
            f"({_display_int(stats.get('read_calls'))}/"
            f"{_display_int(stats.get('write_calls'))}) | "
            f"{_display_int(stats.get('tool_workers'))} | "
            f"n={_display_int(stats.get('pairs'))} pairs | "
            f"{_seconds(stats.get('serial_p50_seconds'))} | "
            f"{_seconds(stats.get('xcode_p50_seconds'))} | "
            f"{_percent(stats.get('p50_latency_reduction'))} | "
            f"{_seconds(stats.get('serial_p95_seconds'))} | "
            f"{_seconds(stats.get('xcode_p95_seconds'))} | "
            f"{_percent(stats.get('p95_latency_reduction'))} | "
            f"{_speedup(stats.get('median_paired_speedup'))} | "
            f"{_display_int(stats.get('xcode_max_concurrency'))} |"
        )
    overall_raw = summary.get("overall")
    overall = overall_raw if isinstance(overall_raw, dict) else {}
    lines.extend(
        [
            "",
            "## Aggregate invariants",
            "",
            (
                "- Median paired latency reduction: "
                f"{_percent(overall.get('median_paired_latency_reduction'))}."
            ),
            (
                "- Median paired speedup: "
                f"{_speedup(overall.get('median_paired_speedup'))}."
            ),
            (
                "- Xcode write-isolation rate: "
                f"{_percent(overall.get('write_isolation_rate'))}."
            ),
            (
                "- Serial/Xcode output equivalence: "
                f"{_percent(overall.get('output_equivalence_rate'))}."
            ),
        ]
    )
    raw_excluded = summary.get("excluded_pairs")
    excluded = raw_excluded if isinstance(raw_excluded, list) else []
    if excluded:
        lines.extend(["", "## Excluded pairs", ""])
        for item in excluded:
            if not isinstance(item, dict):
                continue
            reasons = item.get("reasons")
            reason_values = reasons if isinstance(reasons, list) else []
            lines.append(
                f"- {item.get('task_id')} r{item.get('repeat')}: "
                f"{'; '.join(str(reason) for reason in reason_values)}"
            )
    lines.extend(
        [
            "",
            "Performance rows exclude failed, incomplete, or output-mismatched pairs.",
            "Write isolation requires zero overlap between a write and any other tool.",
            "",
        ]
    )
    return "\n".join(lines)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0 else 0.0


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _display_int(value: object) -> str:
    return (
        str(value) if isinstance(value, int) and not isinstance(value, bool) else "n/a"
    )


def _seconds(value: object) -> str:
    return f"{float(value):.3f}s" if isinstance(value, (int, float)) else "n/a"


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"


def _speedup(value: object) -> str:
    return f"{float(value):.2f}x" if isinstance(value, (int, float)) else "n/a"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate a tool scheduling benchmark report."
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    records = load_tool_scheduling_records(args.results)
    output_dir = args.output_dir or args.results
    write_tool_scheduling_report(records, output_dir.resolve())
    print(output_dir.resolve() / "report.md")


if __name__ == "__main__":
    main()
