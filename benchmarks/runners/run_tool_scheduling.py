"""串行与副作用感知工具调度的配对 benchmark CLI。"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import IO
from uuid import uuid4

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from benchmarks.reports.generate_tool_scheduling_report import (
    write_tool_scheduling_report,
)
from benchmarks.tool_scheduling import (
    SchedulingVariant,
    ToolSchedulingTask,
    discover_scheduling_task_files,
    load_scheduling_task,
    measure_scheduling,
)


def main() -> None:
    args = _parser().parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    task_files = discover_scheduling_task_files(args.tasks)
    tasks = [load_scheduling_task(path) for path in task_files]
    if args.workers is not None:
        tasks = [replace(task, tool_workers=args.workers) for task in tasks]
    output_dir = args.output_dir.resolve()
    records = asyncio.run(_run_all(tasks, args, output_dir))
    summary = write_tool_scheduling_report(records, output_dir)
    report_path = output_dir / "report.md"
    print(report_path)
    if summary.get("excluded_pairs"):
        print(f"invalid scheduling pairs remain; see {report_path}", file=sys.stderr)
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired serial/Xcode tool scheduling benchmarks."
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        type=Path,
        default=[Path("benchmarks/tasks/parallel_reads")],
        help="task.json file or directory containing scheduling tasks",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        help="override every task's Xcode worker limit",
    )
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    return parser


async def _run_all(
    tasks: list[ToolSchedulingTask],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(tasks) * 2 * (args.warmup + args.repeat)
    records: list[dict[str, object]] = []
    with _progress_reporter(total, not args.no_progress) as progress:
        for task in tasks:
            for warmup in range(1, args.warmup + 1):
                for variant in _ordered_variants(warmup):
                    label = f"warmup {warmup}/{args.warmup}"
                    await _run_one(
                        task,
                        variant,
                        warmup,
                        output_dir,
                        keep_workspace=False,
                        measured=False,
                        progress=progress,
                        label=label,
                    )
            for repeat in range(1, args.repeat + 1):
                for variant in _ordered_variants(repeat):
                    record = await _run_one(
                        task,
                        variant,
                        repeat,
                        output_dir,
                        keep_workspace=args.keep_workspaces,
                        measured=True,
                        progress=progress,
                        label=f"repeat {repeat}/{args.repeat}",
                    )
                    records.append(record)
                    print(
                        f"{task.id} {variant} r{repeat}: "
                        f"success={record['success']} "
                        f"duration={_record_number(record, 'duration_seconds'):.3f}s "
                        f"max_concurrency={record['max_concurrency']}",
                        flush=True,
                    )
    return records


async def _run_one(
    task: ToolSchedulingTask,
    variant: SchedulingVariant,
    repeat: int,
    output_dir: Path,
    *,
    keep_workspace: bool,
    measured: bool,
    progress: _SchedulingProgress,
    label: str,
) -> dict[str, object]:
    run_id = f"{task.id}-{variant}-r{repeat}-{uuid4().hex[:8]}"
    workspace = output_dir / "workspaces" / run_id
    progress.start(f"{task.id} {variant} · {label}")
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        measurement = await measure_scheduling(
            task,
            variant,
            repeat=repeat,
            workspace=workspace,
        )
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at,
            "description": task.description,
            "read_calls": sum(op.kind == "read" for op in task.operations),
            "write_calls": sum(op.kind == "write" for op in task.operations),
            "controlled_delay_ms_total": sum(op.delay_ms for op in task.operations),
            **measurement.to_dict(),
        }
    except Exception as exc:
        record = _error_record(task, variant, repeat, run_id, started_at, exc)
    finally:
        if workspace.exists() and (not keep_workspace or not measured):
            shutil.rmtree(workspace)
    if measured:
        _write_record(output_dir / f"{run_id}.json", record)
    progress.finish(
        f"success={record['success']} · "
        f"{_record_number(record, 'duration_seconds'):.3f}s"
    )
    return record


def _error_record(
    task: ToolSchedulingTask,
    variant: SchedulingVariant,
    repeat: int,
    run_id: str,
    started_at: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "task_id": task.id,
        "variant": variant,
        "repeat": repeat,
        "duration_seconds": 0.0,
        "call_count": len(task.operations),
        "completed_calls": 0,
        "failed_calls": len(task.operations),
        "max_concurrency": 0,
        "max_read_concurrency": 0,
        "max_write_concurrency": 0,
        "unsafe_overlap_events": 0,
        "result_order_correct": False,
        "output_digest": "",
        "tool_workers": task.tool_workers,
        "read_calls": sum(op.kind == "read" for op in task.operations),
        "write_calls": sum(op.kind == "write" for op in task.operations),
        "controlled_delay_ms_total": sum(op.delay_ms for op in task.operations),
        "timings": [],
        "success": False,
        "error": str(error),
    }


def _ordered_variants(repeat: int) -> tuple[SchedulingVariant, SchedulingVariant]:
    if repeat % 2 == 0:
        return "xcode", "serial"
    return "serial", "xcode"


class _SchedulingProgress(AbstractContextManager["_SchedulingProgress"]):
    """交互终端进度条和非交互逐行日志的统一外观。"""

    def __init__(self, total: int, enabled: bool) -> None:
        self._console = Console(stderr=True)
        self._stream: IO[str] = self._console.file
        self._enabled = enabled
        self._completed = 0
        self._total = total
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if enabled and self._console.is_terminal:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("tool scheduling"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.fields[status]}", markup=False),
                console=self._console,
            )
            self._task_id = self._progress.add_task(
                "tool scheduling", total=total, status="waiting"
            )

    def __enter__(self) -> _SchedulingProgress:
        if self._progress is not None:
            self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()

    def start(self, label: str) -> None:
        if not self._enabled:
            return
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, status=f"running {label}")
            return
        print(
            f"[{self._completed}/{self._total}] running {label}",
            file=self._stream,
            flush=True,
        )

    def finish(self, detail: str) -> None:
        if not self._enabled:
            return
        self._completed += 1
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id)
            self._progress.update(self._task_id, status=detail)
            return
        print(
            f"[{self._completed}/{self._total}] {detail}",
            file=self._stream,
            flush=True,
        )


def _progress_reporter(total: int, enabled: bool) -> _SchedulingProgress:
    return _SchedulingProgress(total, enabled)


def _write_record(path: Path, record: dict[str, object]) -> None:
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _record_number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmark-results") / "tool_scheduling" / stamp


if __name__ == "__main__":
    main()
