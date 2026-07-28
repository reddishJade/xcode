"""工具并发调度 benchmark 测量内核测试。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from benchmarks.tool_scheduling import (
    discover_scheduling_task_files,
    load_scheduling_task,
    measure_scheduling,
)


def _write_task(
    root: Path,
    operations: list[dict[str, object]],
    *,
    workers: int = 4,
) -> Path:
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    for index in range(1, 5):
        (workspace / f"file-{index}.txt").write_text(
            f"fixture {index}\n", encoding="utf-8"
        )
    manifest = root / "task.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "parallel-read-test",
                "description": "test fixture",
                "workspace": "workspace",
                "tool_workers": workers,
                "operations": operations,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _reads(count: int, delay_ms: int = 20) -> list[dict[str, object]]:
    return [
        {
            "id": f"read-{index}",
            "kind": "read",
            "path": f"file-{index}.txt",
            "delay_ms": delay_ms,
        }
        for index in range(1, count + 1)
    ]


async def test_measure_scheduling_uses_production_parallel_execution(
    tmp_path: Path,
) -> None:
    task = load_scheduling_task(_write_task(tmp_path / "task", _reads(4)))

    serial = await measure_scheduling(
        task, "serial", repeat=1, workspace=tmp_path / "serial"
    )
    xcode = await measure_scheduling(
        task, "xcode", repeat=1, workspace=tmp_path / "xcode"
    )

    assert serial.success
    assert xcode.success
    assert serial.max_concurrency == 1
    assert xcode.max_concurrency == 4
    assert xcode.duration_seconds < serial.duration_seconds * 0.7
    assert xcode.output_digest == serial.output_digest
    assert xcode.workspace_digest == serial.workspace_digest


async def test_mixed_workload_preserves_write_barriers(tmp_path: Path) -> None:
    operations = [
        *_reads(2, delay_ms=5),
        {
            "id": "write-1",
            "kind": "write",
            "path": "output/probe.txt",
            "content": "written\n",
            "delay_ms": 5,
        },
        {
            "id": "read-3",
            "kind": "read",
            "path": "file-3.txt",
            "delay_ms": 5,
        },
    ]
    task = load_scheduling_task(_write_task(tmp_path / "task", operations))

    serial = await measure_scheduling(
        task, "serial", repeat=1, workspace=tmp_path / "serial"
    )
    result = await measure_scheduling(
        task, "xcode", repeat=1, workspace=tmp_path / "run"
    )

    assert result.success
    assert result.max_read_concurrency == 2
    assert result.max_write_concurrency == 1
    assert result.unsafe_overlap_events == 0
    assert result.workspace_digest == serial.workspace_digest
    assert (tmp_path / "run/output/probe.txt").read_text(encoding="utf-8") == (
        "written\n"
    )


def test_load_scheduling_task_rejects_escaping_operation_path(
    tmp_path: Path,
) -> None:
    manifest = _write_task(
        tmp_path / "task",
        [{"id": "read-1", "kind": "read", "path": "../outside.txt"}],
    )

    with pytest.raises(ValueError, match="path must stay relative"):
        load_scheduling_task(manifest)


def test_included_scheduling_tasks_match_schema() -> None:
    repository = Path(__file__).resolve().parents[3]
    task_root = repository / "benchmarks/tasks/parallel_reads"
    schema = json.loads((task_root / "task.schema.json").read_text(encoding="utf-8"))
    manifests = discover_scheduling_task_files([task_root])

    assert len(manifests) == 4
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        jsonschema.validate(payload, schema)
        load_scheduling_task(manifest)
