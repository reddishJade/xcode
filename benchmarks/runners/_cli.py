"""长程 benchmark 命令行公共逻辑。"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.models import discover_task_files, load_task
from benchmarks.reports.generate_report import write_report
from benchmarks.runners._long_horizon import RunOptions, Variant, run_task
from benchmarks.runners.progress import create_progress_reporter
from xcode.harness.config import (
    XcodeRuntimeConfig,
    discover_runtime_config,
    load_runtime_config,
)


def run_variant_main(variant: Variant) -> None:
    """运行单个消融分组。"""
    parser = _parser(f"Run the {variant} long-horizon benchmark variant.")
    args = parser.parse_args()
    records = _run_variants(args, (variant,))
    _write_and_validate_report(records, args, (variant,))


def run_ablation_main() -> None:
    """交替顺序运行 baseline 与 Xcode，并立即生成配对报告。"""
    parser = _parser("Run paired baseline/Xcode long-horizon ablations.")
    args = parser.parse_args()
    records = _run_variants(args, ("baseline", "xcode"))
    _write_and_validate_report(records, args, ("baseline", "xcode"))


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "tasks",
        nargs="*",
        type=Path,
        default=[Path("benchmarks/tasks/long_horizon")],
        help="task.json file or a directory containing task.json files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="explicit Xcode runtime config; otherwise discover config from cwd",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--max-pair-attempts",
        type=int,
        default=1,
        help=(
            "rerun a baseline/Xcode pair after transient incomplete usage; "
            "original attempts are retained"
        ),
    )
    parser.add_argument(
        "--require-complete-usage",
        "--fail-on-incomplete",
        dest="require_complete_usage",
        action="store_true",
        help="write the report, then exit nonzero if a primary usage cohort is incomplete",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the progress bar and progress log output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
    )
    return parser


def _run_variants(
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
) -> list[dict[str, object]]:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.max_pair_attempts <= 0:
        raise ValueError("--max-pair-attempts must be positive")
    runtime_config = _runtime_config(args.config)
    task_files = discover_task_files(args.tasks)
    tasks = [load_task(path) for path in task_files]
    records: list[dict[str, object]] = []
    total_runs = args.repeat * len(tasks) * len(variants)
    reporter = create_progress_reporter(
        total_runs,
        enabled=not args.no_progress,
    )
    with reporter:
        for repetition in range(1, args.repeat + 1):
            ordered = variants
            if len(variants) == 2 and repetition % 2 == 0:
                ordered = tuple(reversed(variants))
            for task in tasks:
                for attempt in range(1, args.max_pair_attempts + 1):
                    attempt_records: list[dict[str, object]] = []
                    for variant in ordered:
                        options = RunOptions(
                            output_dir=args.output_dir.resolve(),
                            repeat=repetition,
                            attempt=attempt,
                            temperature=args.temperature,
                            keep_workspace=args.keep_workspaces,
                            progress_callback=reporter.update,
                        )
                        record = run_task(task, variant, runtime_config, options)
                        attempt_records.append(record)
                        records.append(record)
                        attempt_label = f" a{attempt}" if attempt > 1 else ""
                        print(
                            f"{task.id} {variant} r{repetition}{attempt_label}: "
                            f"success={record['task_success']} "
                            f"usage_complete={record['usage_complete']} "
                            f"input_tokens={record['input_tokens_total']}"
                        )
                    if all(
                        bool(record.get("usage_complete")) for record in attempt_records
                    ):
                        break
                    if attempt >= args.max_pair_attempts or not _retryable_attempt(
                        attempt_records
                    ):
                        break
                    reporter.add_runs(len(variants))
                    reason = _attempt_issue_detail(attempt_records)
                    print(
                        f"{task.id} r{repetition}: retrying full pair as "
                        f"attempt {attempt + 1}/{args.max_pair_attempts}: {reason}",
                        file=sys.stderr,
                        flush=True,
                    )
    return records


def _retryable_attempt(records: list[dict[str, object]]) -> bool:
    """仅对具有明确瞬时 provider 错误的 usage 缺失执行重跑。"""
    incomplete = [record for record in records if not record.get("usage_complete")]
    return bool(incomplete) and all(
        bool(record.get("retryable_usage_failure")) for record in incomplete
    )


def _attempt_issue_detail(records: list[dict[str, object]]) -> str:
    details: list[str] = []
    for record in records:
        if record.get("usage_complete"):
            continue
        variant = str(record.get("variant", "run"))
        raw_issues = record.get("usage_incomplete_calls")
        issues = raw_issues if isinstance(raw_issues, list) else []
        errors = [
            str(issue.get("error", "missing usage"))
            for issue in issues
            if isinstance(issue, dict)
        ]
        details.append(f"{variant}: {', '.join(errors) or 'missing usage'}")
    return "; ".join(details) or "incomplete provider usage"


def _write_and_validate_report(
    records: list[dict[str, object]],
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
) -> None:
    output_dir = args.output_dir.resolve()
    summary = write_report(records, output_dir)
    report_path = output_dir / "report.md"
    print(report_path)
    if not args.require_complete_usage:
        return
    if len(variants) == 2:
        cohorts = summary.get("cohorts")
        values = cohorts if isinstance(cohorts, dict) else {}
        complete = int(values.get("complete_usage_pairs", 0))
        expected = int(summary.get("paired_runs", 0))
        is_incomplete = complete < expected
    else:
        raw_variant = summary.get("variants")
        variant_values = raw_variant if isinstance(raw_variant, dict) else {}
        raw_stats = variant_values.get(variants[0])
        stats = raw_stats if isinstance(raw_stats, dict) else {}
        is_incomplete = int(stats.get("usage_complete_runs", 0)) < int(
            stats.get("runs", 0)
        )
    if is_incomplete:
        print(
            f"incomplete provider usage remains; see {report_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _runtime_config(path: Path | None) -> XcodeRuntimeConfig:
    if path is not None:
        return load_runtime_config(path.resolve())
    return discover_runtime_config(Path.cwd())


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark-results") / "long_horizon" / stamp
