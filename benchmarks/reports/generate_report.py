"""从原始运行记录生成配对消融报告。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

_VARIANTS = ("baseline", "xcode")
_PHASES = ("pre_compaction", "post_compaction", "post_resume")


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
    """按逻辑配对选择主 attempt，并为每项指标建立一致 cohort。"""
    paired_attempts: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    variant_attempts: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        variant = str(record.get("variant", ""))
        if variant not in _VARIANTS:
            continue
        task_id = str(record.get("task_id", ""))
        repeat = int(record.get("repeat", 0))
        attempt = _attempt_number(record)
        key = (task_id, repeat, attempt)
        if variant in paired_attempts[key]:
            raise ValueError(
                "duplicate run for paired attempt "
                f"task={task_id!r}, repeat={repeat}, attempt={attempt}, "
                f"variant={variant!r}"
            )
        paired_attempts[key][variant] = record
        variant_attempts[(task_id, repeat, variant)].append(record)

    complete_pair_attempts = [
        pair for pair in paired_attempts.values() if len(pair) == 2
    ]
    for pair in complete_pair_attempts:
        _validate_pair_controls(pair)
    logical_pairs: dict[tuple[str, int], list[dict[str, dict[str, Any]]]] = defaultdict(
        list
    )
    for (task_id, repeat, _attempt), pair in paired_attempts.items():
        if len(pair) == 2:
            logical_pairs[(task_id, repeat)].append(pair)
    selected_pairs = [
        _select_primary_pair(logical_pairs[key]) for key in sorted(logical_pairs)
    ]

    selected_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired_keys = set(logical_pairs)
    for pair in selected_pairs:
        for variant in _VARIANTS:
            selected_by_variant[variant].append(pair[variant])
    for (task_id, repeat, variant), attempts in variant_attempts.items():
        if (task_id, repeat) in paired_keys:
            continue
        selected_by_variant[variant].append(_select_primary_record(attempts))

    complete_usage_pairs = [
        pair for pair in selected_pairs if _pair_usage_complete(pair)
    ]
    phase_pairs = {
        phase: [
            pair for pair in selected_pairs if _pair_phase_usage_complete(pair, phase)
        ]
        for phase in _PHASES
    }
    usage_by_variant = {
        variant: [pair[variant] for pair in complete_usage_pairs]
        for variant in _VARIANTS
    }
    for variant in _VARIANTS:
        usage_by_variant[variant].extend(
            record
            for record in selected_by_variant.get(variant, [])
            if (
                str(record.get("task_id", "")),
                int(record.get("repeat", 0)),
            )
            not in paired_keys
            and record.get("usage_complete")
        )
    variants = {
        variant: _variant_summary(
            selected_by_variant.get(variant, []),
            usage_by_variant[variant],
        )
        for variant in _VARIANTS
    }
    for phase, pairs in phase_pairs.items():
        field = f"{phase}_input_tokens"
        for variant in _VARIANTS:
            variants[variant][f"{field}_mean"] = _mean_field(
                [pair[variant] for pair in pairs],
                field,
            )
    excluded_attempts = [
        _excluded_attempt(pair)
        for pair in complete_pair_attempts
        if not _pair_usage_complete(pair)
    ]
    return {
        "schema_version": 2,
        "runs": len(records),
        "primary_runs": sum(len(values) for values in selected_by_variant.values()),
        "paired_runs": len(selected_pairs),
        "pair_attempts": len(complete_pair_attempts),
        "retried_pairs": sum(len(attempts) > 1 for attempts in logical_pairs.values()),
        "cohorts": {
            "correctness_pairs": len(selected_pairs),
            "complete_usage_pairs": len(complete_usage_pairs),
            **{
                f"{phase}_usage_pairs": len(pairs)
                for phase, pairs in phase_pairs.items()
            },
        },
        "variants": variants,
        "paired_changes": _paired_changes(
            selected_pairs,
            complete_usage_pairs,
            phase_pairs,
        ),
        "selected_pairs": [_selected_pair_summary(pair) for pair in selected_pairs],
        "excluded_attempts": excluded_attempts,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """将聚合结果渲染为适合 README/简历取数的 Markdown。"""
    variants = summary["variants"]
    baseline = variants["baseline"]
    xcode = variants["xcode"]
    changes = summary["paired_changes"]
    cohorts = summary["cohorts"]
    correctness_cohort = _cohort(cohorts.get("correctness_pairs"))
    usage_cohort = _cohort(cohorts.get("complete_usage_pairs"))
    pre_cohort = _cohort(cohorts.get("pre_compaction_usage_pairs"))
    post_cohort = _cohort(cohorts.get("post_compaction_usage_pairs"))
    resume_cohort = _cohort(cohorts.get("post_resume_usage_pairs"))
    lines = [
        "# Long-horizon compaction ablation",
        "",
        (
            f"Runs: {summary['runs']} attempt records, "
            f"{summary['pair_attempts']} pair attempts, "
            f"{summary['paired_runs']} selected task/repetition pairs."
        ),
        f"Retried logical pairs: {summary['retried_pairs']}.",
        "",
        "| Metric | Cohort | Baseline | Xcode | Paired change |",
        "|---|---:|---:|---:|---:|",
        _row(
            "Mean input tokens",
            usage_cohort,
            baseline.get("input_tokens_mean"),
            xcode.get("input_tokens_mean"),
            changes.get("input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Peak input tokens",
            usage_cohort,
            baseline.get("peak_input_tokens_mean"),
            xcode.get("peak_input_tokens_mean"),
            changes.get("peak_input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Pre-compaction input tokens",
            pre_cohort,
            baseline.get("pre_compaction_input_tokens_mean"),
            xcode.get("pre_compaction_input_tokens_mean"),
            changes.get("pre_compaction_input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Post-compaction input tokens",
            post_cohort,
            baseline.get("post_compaction_input_tokens_mean"),
            xcode.get("post_compaction_input_tokens_mean"),
            changes.get("post_compaction_input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Post-resume input tokens",
            resume_cohort,
            baseline.get("post_resume_input_tokens_mean"),
            xcode.get("post_resume_input_tokens_mean"),
            changes.get("post_resume_input_token_reduction"),
            change_kind="percent",
        ),
        _row(
            "Input cost (USD)",
            usage_cohort,
            baseline.get("input_cost_mean"),
            xcode.get("input_cost_mean"),
            changes.get("input_cost_reduction"),
            value_kind="cost",
            change_kind="percent",
        ),
        _row(
            "Duration (seconds)",
            correctness_cohort,
            baseline.get("duration_seconds_mean"),
            xcode.get("duration_seconds_mean"),
            changes.get("duration_reduction"),
            change_kind="percent",
        ),
        _row(
            "Task success rate",
            correctness_cohort,
            baseline.get("task_success_rate"),
            xcode.get("task_success_rate"),
            changes.get("task_success_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        _row(
            "Long-session completion rate",
            correctness_cohort,
            baseline.get("long_session_completion_rate"),
            xcode.get("long_session_completion_rate"),
            changes.get("long_session_completion_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        _row(
            "State retention",
            correctness_cohort,
            baseline.get("state_retention_mean"),
            xcode.get("state_retention_mean"),
            changes.get("state_retention_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        _row(
            "Context overflow rate",
            correctness_cohort,
            baseline.get("context_overflow_rate"),
            xcode.get("context_overflow_rate"),
            changes.get("context_overflow_change_pp"),
            value_kind="percent",
            change_kind="points",
        ),
        "",
        (
            "Post-compaction includes the compaction-summary request; post-resume "
            "starts on the turn after the declared restart boundary."
        ),
        "Each metric excludes only pairs whose provider usage is incomplete for that metric.",
        "Task success is determined by the task verification command, not model self-report.",
    ]
    excluded = summary.get("excluded_attempts")
    if isinstance(excluded, list) and excluded:
        lines.extend(["", "## Excluded attempts", ""])
        for item in excluded:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('task_id')} r{item.get('repeat')} "
                f"a{item.get('attempt')}: {item.get('reason')}"
            )
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


def _variant_summary(
    records: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "usage_complete_runs": sum(
            bool(record.get("usage_complete")) for record in records
        ),
        "token_cohort_runs": len(usage_records),
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
    selected_pairs: list[dict[str, dict[str, Any]]],
    complete_usage_pairs: list[dict[str, dict[str, Any]]],
    phase_pairs: dict[str, list[dict[str, dict[str, Any]]]],
) -> dict[str, float | None]:
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
        **{
            f"{phase}_input_token_reduction": _paired_reduction(
                pairs,
                f"{phase}_input_tokens",
            )
            for phase, pairs in phase_pairs.items()
        },
        "duration_reduction": _paired_reduction(
            selected_pairs,
            "duration_seconds",
        ),
        "task_success_change_pp": _paired_point_change(selected_pairs, "task_success"),
        "long_session_completion_change_pp": _paired_point_change(
            selected_pairs, "long_session_completed"
        ),
        "state_retention_change_pp": _paired_numeric_point_change(
            selected_pairs, "state_retention"
        ),
        "context_overflow_change_pp": _paired_point_change(
            selected_pairs, "context_overflow"
        ),
    }


def _validate_pair_controls(pair: dict[str, dict[str, Any]]) -> None:
    baseline = pair["baseline"]
    xcode = pair["xcode"]
    for field in (
        "model",
        "temperature",
        "execution_mode",
        "summary_mode",
        "baseline_commit",
    ):
        if baseline.get(field) != xcode.get(field):
            raise ValueError(
                f"paired runs differ on control {field}: "
                f"{baseline.get(field)!r} != {xcode.get(field)!r}"
            )


def _attempt_number(record: dict[str, Any]) -> int:
    value = record.get("attempt", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid benchmark attempt: {value!r}")
    return value


def _select_primary_pair(
    pairs: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    ordered = sorted(pairs, key=lambda pair: _attempt_number(pair["baseline"]))
    return next((pair for pair in ordered if _pair_usage_complete(pair)), ordered[-1])


def _select_primary_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=_attempt_number)
    return next(
        (record for record in ordered if record.get("usage_complete")),
        ordered[-1],
    )


def _pair_usage_complete(pair: dict[str, dict[str, Any]]) -> bool:
    return all(bool(pair[variant].get("usage_complete")) for variant in _VARIANTS)


def _pair_phase_usage_complete(
    pair: dict[str, dict[str, Any]],
    phase: str,
) -> bool:
    return all(
        bool(pair[variant].get(f"{phase}_usage_complete")) for variant in _VARIANTS
    )


def _selected_pair_summary(
    pair: dict[str, dict[str, Any]],
) -> dict[str, object]:
    baseline = pair["baseline"]
    return {
        "task_id": str(baseline.get("task_id", "")),
        "repeat": int(baseline.get("repeat", 0)),
        "attempt": _attempt_number(baseline),
        "usage_complete": _pair_usage_complete(pair),
    }


def _excluded_attempt(
    pair: dict[str, dict[str, Any]],
) -> dict[str, object]:
    baseline = pair["baseline"]
    reasons: list[str] = []
    for variant in _VARIANTS:
        record = pair[variant]
        if record.get("usage_complete"):
            continue
        raw_issues = record.get("usage_incomplete_calls")
        issues = raw_issues if isinstance(raw_issues, list) else []
        errors = [
            str(issue.get("error", "missing usage"))
            for issue in issues
            if isinstance(issue, dict)
        ]
        reasons.append(f"{variant}: {', '.join(errors) or 'provider usage incomplete'}")
    return {
        "task_id": str(baseline.get("task_id", "")),
        "repeat": int(baseline.get("repeat", 0)),
        "attempt": _attempt_number(baseline),
        "reason": "; ".join(reasons),
    }


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
    cohort: str,
    baseline: object,
    xcode: object,
    change: object,
    *,
    value_kind: str = "number",
    change_kind: str = "number",
) -> str:
    return (
        f"| {label} | {cohort} | {_format_value(baseline, value_kind)} | "
        f"{_format_value(xcode, value_kind)} | "
        f"{_format_value(change, change_kind)} |"
    )


def _cohort(value: object) -> str:
    count = value if isinstance(value, int) and not isinstance(value, bool) else 0
    return f"n={count} pairs"


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
