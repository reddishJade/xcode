"""从原始运行记录生成配对消融报告。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

_VARIANTS = ("baseline", "xcode")


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    """读取文件或目录中的单次运行 JSON。"""
    files: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(
                item
                for item in path.glob("*.json")
                if item.name not in {"summary.json"}
            )
        else:
            raise ValueError(f"record path does not exist: {path}")
    records: list[dict[str, Any]] = []
    for path in sorted(files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("variant") in _VARIANTS:
            records.append({str(key): value for key, value in payload.items()})
    if not records:
        raise ValueError("no benchmark run records found")
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合全部运行，并仅用同 task/repeat 的记录计算变化。"""
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        variant = str(record.get("variant", ""))
        if variant not in _VARIANTS:
            continue
        by_variant[variant].append(record)
        key = (str(record.get("task_id", "")), int(record.get("repeat", 0)))
        if variant in paired[key]:
            raise ValueError(
                "duplicate run for paired key "
                f"task={key[0]!r}, repeat={key[1]}, variant={variant!r}"
            )
        paired[key][variant] = record
    pairs = [value for value in paired.values() if len(value) == 2]
    for pair in pairs:
        _validate_pair_controls(pair)
    variants = {
        variant: _variant_summary(by_variant.get(variant, [])) for variant in _VARIANTS
    }
    return {
        "schema_version": 1,
        "runs": len(records),
        "paired_runs": len(pairs),
        "variants": variants,
        "paired_changes": _paired_changes(pairs),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """将聚合结果渲染为适合 README/简历取数的 Markdown。"""
    variants = summary["variants"]
    baseline = variants["baseline"]
    xcode = variants["xcode"]
    changes = summary["paired_changes"]
    lines = [
        "# Long-horizon compaction ablation",
        "",
        (
            f"Runs: {summary['runs']} total, {summary['paired_runs']} paired "
            "task/repetition samples."
        ),
        "",
        "| Metric | Baseline | Xcode | Paired change |",
        "|---|---:|---:|---:|",
        _row(
            "Mean input tokens",
            baseline.get("input_tokens_mean"),
            xcode.get("input_tokens_mean"),
            changes.get("input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Peak input tokens",
            baseline.get("peak_input_tokens_mean"),
            xcode.get("peak_input_tokens_mean"),
            changes.get("peak_input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Input cost (USD)",
            baseline.get("input_cost_mean"),
            xcode.get("input_cost_mean"),
            changes.get("input_cost_reduction"),
            value_kind="cost",
            change_kind="percent",
        ),
        _row(
            "Task success rate",
            baseline.get("task_success_rate"),
            xcode.get("task_success_rate"),
            changes.get("task_success_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        _row(
            "Long-session completion rate",
            baseline.get("long_session_completion_rate"),
            xcode.get("long_session_completion_rate"),
            changes.get("long_session_completion_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        _row(
            "State retention",
            baseline.get("state_retention_mean"),
            xcode.get("state_retention_mean"),
            changes.get("state_retention_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        _row(
            "Context overflow rate",
            baseline.get("context_overflow_rate"),
            xcode.get("context_overflow_rate"),
            changes.get("context_overflow_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        "",
        "Token and cost rows exclude runs whose provider did not return complete usage.",
        "Task success is determined by the task verification command, not model self-report.",
    ]
    return "\n".join(lines) + "\n"


def write_report(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """同时写出机器可读摘要和 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _variant_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    usage_records = [record for record in records if record.get("usage_complete")]
    retentions = [
        float(value)
        for record in records
        if (value := record.get("state_retention")) is not None
    ]
    costs = [
        float(value)
        for record in usage_records
        if (value := record.get("input_cost_usd")) is not None
    ]
    durations = [float(record.get("duration_seconds", 0)) for record in records]
    return {
        "runs": len(records),
        "usage_complete_runs": len(usage_records),
        "input_tokens_mean": _mean_field(usage_records, "input_tokens_total"),
        "input_tokens_median": _median_field(usage_records, "input_tokens_total"),
        "peak_input_tokens_mean": _mean_field(usage_records, "peak_input_tokens"),
        "input_cost_mean": fmean(costs) if costs else None,
        "task_success_rate": _boolean_rate(records, "task_success"),
        "long_session_completion_rate": _boolean_rate(
            records, "long_session_completed"
        ),
        "state_retention_mean": fmean(retentions) if retentions else None,
        "context_overflow_rate": _boolean_rate(records, "context_overflow"),
        "duration_seconds_mean": fmean(durations) if durations else None,
    }


def _paired_changes(
    pairs: list[dict[str, dict[str, Any]]],
) -> dict[str, float | None]:
    complete_usage_pairs = [
        pair
        for pair in pairs
        if pair["baseline"].get("usage_complete")
        and pair["xcode"].get("usage_complete")
    ]
    return {
        "input_token_reduction": _paired_reduction(
            complete_usage_pairs, "input_tokens_total"
        ),
        "peak_input_token_reduction": _paired_reduction(
            complete_usage_pairs, "peak_input_tokens"
        ),
        "input_cost_reduction": _paired_reduction(
            [
                pair
                for pair in complete_usage_pairs
                if pair["baseline"].get("input_cost_usd") is not None
                and pair["xcode"].get("input_cost_usd") is not None
            ],
            "input_cost_usd",
        ),
        "task_success_change_pp": _paired_point_change(pairs, "task_success"),
        "long_session_completion_change_pp": _paired_point_change(
            pairs, "long_session_completed"
        ),
        "state_retention_change_pp": _paired_numeric_point_change(
            pairs, "state_retention"
        ),
        "context_overflow_change_pp": _paired_point_change(pairs, "context_overflow"),
    }


def _validate_pair_controls(pair: dict[str, dict[str, Any]]) -> None:
    baseline = pair["baseline"]
    xcode = pair["xcode"]
    for field in ("model", "temperature", "execution_mode"):
        if baseline.get(field) != xcode.get(field):
            raise ValueError(
                f"paired runs differ on control {field}: "
                f"{baseline.get(field)!r} != {xcode.get(field)!r}"
            )


def _paired_reduction(
    pairs: list[dict[str, dict[str, Any]]], field: str
) -> float | None:
    if not pairs:
        return None
    baseline = fmean(float(pair["baseline"].get(field, 0)) for pair in pairs)
    xcode = fmean(float(pair["xcode"].get(field, 0)) for pair in pairs)
    if baseline == 0:
        return None
    return (baseline - xcode) / baseline


def _paired_point_change(
    pairs: list[dict[str, dict[str, Any]]], field: str
) -> float | None:
    if not pairs:
        return None
    differences = [
        float(bool(pair["xcode"].get(field))) - float(bool(pair["baseline"].get(field)))
        for pair in pairs
    ]
    return fmean(differences)


def _paired_numeric_point_change(
    pairs: list[dict[str, dict[str, Any]]], field: str
) -> float | None:
    differences = [
        float(pair["xcode"][field]) - float(pair["baseline"][field])
        for pair in pairs
        if pair["xcode"].get(field) is not None
        and pair["baseline"].get(field) is not None
    ]
    return fmean(differences) if differences else None


def _mean_field(records: list[dict[str, Any]], field: str) -> float | None:
    return fmean(float(record.get(field, 0)) for record in records) if records else None


def _median_field(records: list[dict[str, Any]], field: str) -> float | None:
    return (
        float(median(float(record.get(field, 0)) for record in records))
        if records
        else None
    )


def _boolean_rate(records: list[dict[str, Any]], field: str) -> float | None:
    if not records:
        return None
    return fmean(float(bool(record.get(field))) for record in records)


def _row(
    label: str,
    baseline: object,
    xcode: object,
    change: object,
    *,
    value_kind: str = "number",
    change_kind: str = "number",
) -> str:
    return (
        f"| {label} | {_format_value(baseline, value_kind)} | "
        f"{_format_value(xcode, value_kind)} | "
        f"{_format_value(change, change_kind)} |"
    )


def _format_value(value: object, kind: str) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    number = float(value)
    if kind == "percent":
        return f"{number * 100:.1f}%"
    if kind == "points":
        return f"{number * 100:+.1f} pp"
    if kind == "cost":
        return f"${number:.4f}"
    return f"{number:,.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.paths)
    write_report(records, args.output_dir.resolve())
    print(args.output_dir.resolve() / "report.md")


if __name__ == "__main__":
    main()
