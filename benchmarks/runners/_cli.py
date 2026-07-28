"""长程 benchmark 命令行公共逻辑。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from xcode.harness.config import (
    XcodeRuntimeConfig,
    discover_runtime_config,
    load_runtime_config,
)

from benchmarks.models import discover_task_files, load_task
from benchmarks.reports.generate_report import write_report
from benchmarks.runners._long_horizon import RunOptions, SummaryMode, Variant, run_task
from benchmarks.runners.progress import create_progress_reporter


def run_variant_main(variant: Variant) -> None:
    """运行单个消融分组。"""
    parser = _parser(f"Run the {variant} long-horizon benchmark variant.")
    args = parser.parse_args()
    records = _run_variants(args, (variant,))
    write_report(records, args.output_dir.resolve())
    print(args.output_dir.resolve() / "report.md")


def run_ablation_main() -> None:
    """交替顺序运行 baseline 与 Xcode，并立即生成配对报告。"""
    parser = _parser("Run paired baseline/Xcode long-horizon ablations.")
    args = parser.parse_args()
    records = _run_variants(args, ("baseline", "xcode"))
    write_report(records, args.output_dir.resolve())
    print(args.output_dir.resolve() / "report.md")


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
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--summary-mode",
        choices=("model", "deterministic"),
        default="model",
    )
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
                for variant in ordered:
                    options = RunOptions(
                        output_dir=args.output_dir.resolve(),
                        repeat=repetition,
                        temperature=args.temperature,
                        summary_mode=_summary_mode(args.summary_mode),
                        keep_workspace=args.keep_workspaces,
                        progress_callback=reporter.update,
                    )
                    record = run_task(task, variant, runtime_config, options)
                    records.append(record)
                    print(
                        f"{task.id} {variant} r{repetition}: "
                        f"success={record['task_success']} "
                        f"input_tokens={record['input_tokens_total']}"
                    )
    return records


def _runtime_config(path: Path | None) -> XcodeRuntimeConfig:
    if path is not None:
        return load_runtime_config(path.resolve())
    return discover_runtime_config(Path.cwd())


def _summary_mode(value: str) -> SummaryMode:
    if value == "deterministic":
        return "deterministic"
    return "model"


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark-results") / "long_horizon" / stamp
