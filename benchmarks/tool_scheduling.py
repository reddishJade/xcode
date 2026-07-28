"""工具调度 benchmark 的任务模型与确定性测量内核。"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Literal, Mapping, cast

from xcode.agent._execution import execute_tool_calls
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.events import AgentEvent
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import (
    AgentToolResult,
    CancellationSignal,
    TextContent,
    ToolArguments,
    ToolCallContent,
    ToolExecutionMode,
    ToolUpdateCallback,
)

SchedulingVariant = Literal["serial", "xcode"]
OperationKind = Literal["read", "write"]

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SchedulingOperation:
    """一次具有明确副作用分类的工具调用。"""

    id: str
    kind: OperationKind
    path: str
    delay_ms: float = 0.0
    content: str = ""


@dataclass(frozen=True)
class ToolSchedulingTask:
    """一组由生产调度器执行的确定性工具调用。"""

    schema_version: int
    id: str
    description: str
    manifest_path: Path
    workspace: Path
    tool_workers: int
    operations: tuple[SchedulingOperation, ...]


@dataclass(frozen=True)
class ToolTiming:
    """单次工具调用相对于批次起点的时间信息。"""

    call_id: str
    kind: OperationKind
    started_offset_seconds: float
    finished_offset_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class SchedulingMeasurement:
    """一次串行或安全并发调度的原始测量结果。"""

    task_id: str
    variant: SchedulingVariant
    repeat: int
    duration_seconds: float
    call_count: int
    completed_calls: int
    failed_calls: int
    max_concurrency: int
    max_read_concurrency: int
    max_write_concurrency: int
    unsafe_overlap_events: int
    result_order_correct: bool
    output_digest: str
    tool_workers: int
    timings: tuple[ToolTiming, ...]

    @property
    def success(self) -> bool:
        return (
            self.completed_calls == self.call_count
            and self.failed_calls == 0
            and self.unsafe_overlap_events == 0
            and self.result_order_correct
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["success"] = self.success
        return value


def load_scheduling_task(path: Path) -> ToolSchedulingTask:
    """读取工具调度任务，并拒绝路径逃逸和未知字段。"""
    manifest_path = path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = _mapping(payload, "task manifest")
    _reject_unknown(
        data,
        {
            "schema_version",
            "id",
            "description",
            "workspace",
            "tool_workers",
            "operations",
        },
        "task manifest",
    )
    schema_version = _integer(data.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    task_id = _text(data.get("id"), "id")
    if not _SAFE_TASK_ID.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id!r}")
    workspace_value = _safe_relative_path(_text(data.get("workspace"), "workspace"))
    workspace = (manifest_path.parent / workspace_value).resolve()
    _ensure_within(manifest_path.parent.resolve(), workspace, "workspace")
    if not workspace.is_dir():
        raise ValueError(f"task workspace does not exist: {workspace}")
    tool_workers = _positive_integer(data.get("tool_workers", 4), "tool_workers")
    raw_operations = _list(data.get("operations"), "operations")
    if not raw_operations:
        raise ValueError("operations must not be empty")
    operations = tuple(
        _load_operation(value, index) for index, value in enumerate(raw_operations, 1)
    )
    operation_ids = [operation.id for operation in operations]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("operation ids must be unique")
    for operation in operations:
        operation_path = (workspace / operation.path).resolve()
        _ensure_within(workspace, operation_path, f"operation {operation.id!r} path")
        if operation.kind == "read" and not operation_path.is_file():
            raise ValueError(
                f"read operation {operation.id!r} file does not exist: {operation_path}"
            )
    return ToolSchedulingTask(
        schema_version=schema_version,
        id=task_id,
        description=_text(data.get("description", task_id), "description"),
        manifest_path=manifest_path,
        workspace=workspace,
        tool_workers=tool_workers,
        operations=operations,
    )


def discover_scheduling_task_files(paths: list[Path]) -> tuple[Path, ...]:
    """展开清单文件或目录，并稳定返回发现的 task.json。"""
    discovered: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file():
            discovered.add(path)
        elif path.is_dir():
            discovered.update(item.resolve() for item in path.rglob("task.json"))
        else:
            raise ValueError(f"task path does not exist: {path}")
    if not discovered:
        raise ValueError("no scheduling task manifests found")
    return tuple(sorted(discovered))


async def measure_scheduling(
    task: ToolSchedulingTask,
    variant: SchedulingVariant,
    *,
    repeat: int,
    workspace: Path,
) -> SchedulingMeasurement:
    """在隔离工作区中通过生产调度器执行一次任务。"""
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    workspace = workspace.resolve()
    if workspace.exists():
        raise ValueError(f"benchmark workspace already exists: {workspace}")
    shutil.copytree(task.workspace, workspace)
    tracker = _ConcurrencyTracker()
    tools = [
        _BenchmarkTool("benchmark_read", "parallel", workspace, tracker),
        _BenchmarkTool("benchmark_write", "sequential", workspace, tracker),
    ]
    tool_calls = [_tool_call(operation) for operation in task.operations]
    assistant_message = AssistantMessage(content=list(tool_calls))
    context = AgentContext(tools=cast(list, tools))
    config = AgentLoopConfig(
        tool_execution="sequential" if variant == "serial" else "parallel",
        tool_workers=task.tool_workers,
        tool_timeout_seconds=max(
            5.0,
            sum(operation.delay_ms for operation in task.operations) / 1_000 + 5,
        ),
    )
    started = time.perf_counter()
    tracker.start_batch(started)
    batch = await execute_tool_calls(
        context,
        assistant_message,
        tool_calls,
        config,
        None,
        _ignore_event,
    )
    duration = time.perf_counter() - started
    result_ids = [result.tool_call_id for result in batch.results]
    expected_ids = [operation.id for operation in task.operations]
    failed_calls = sum(result.is_error for result in batch.results)
    digest = hashlib.sha256()
    for result in batch.results:
        digest.update(result.tool_call_id.encode())
        digest.update(b"\0")
        digest.update(str(result.content).encode())
        digest.update(b"\0")
    return SchedulingMeasurement(
        task_id=task.id,
        variant=variant,
        repeat=repeat,
        duration_seconds=duration,
        call_count=len(tool_calls),
        completed_calls=len(batch.results),
        failed_calls=failed_calls,
        max_concurrency=tracker.max_total,
        max_read_concurrency=tracker.max_reads,
        max_write_concurrency=tracker.max_writes,
        unsafe_overlap_events=tracker.unsafe_overlap_events,
        result_order_correct=result_ids == expected_ids,
        output_digest=digest.hexdigest(),
        tool_workers=task.tool_workers,
        timings=tuple(tracker.timings),
    )


class _ConcurrencyTracker:
    """记录单事件循环内的活动工具数和写入重叠。"""

    def __init__(self) -> None:
        self.batch_started = 0.0
        self.active_total = 0
        self.active_reads = 0
        self.active_writes = 0
        self.max_total = 0
        self.max_reads = 0
        self.max_writes = 0
        self.unsafe_overlap_events = 0
        self._started: dict[str, tuple[OperationKind, float]] = {}
        self.timings: list[ToolTiming] = []

    def start_batch(self, started: float) -> None:
        self.batch_started = started

    def enter(self, call_id: str, kind: OperationKind) -> None:
        now = time.perf_counter()
        if kind == "write" and self.active_total > 0:
            self.unsafe_overlap_events += 1
        if kind == "read" and self.active_writes > 0:
            self.unsafe_overlap_events += 1
        self.active_total += 1
        if kind == "read":
            self.active_reads += 1
        else:
            self.active_writes += 1
        self.max_total = max(self.max_total, self.active_total)
        self.max_reads = max(self.max_reads, self.active_reads)
        self.max_writes = max(self.max_writes, self.active_writes)
        self._started[call_id] = (kind, now)

    def leave(self, call_id: str) -> None:
        kind, started = self._started.pop(call_id)
        finished = time.perf_counter()
        self.active_total -= 1
        if kind == "read":
            self.active_reads -= 1
        else:
            self.active_writes -= 1
        self.timings.append(
            ToolTiming(
                call_id=call_id,
                kind=kind,
                started_offset_seconds=started - self.batch_started,
                finished_offset_seconds=finished - self.batch_started,
                duration_seconds=finished - started,
            )
        )


class _BenchmarkTool:
    """带真实文件 I/O 和可控等待的 benchmark 专用工具。"""

    def __init__(
        self,
        name: str,
        execution_mode: ToolExecutionMode,
        workspace: Path,
        tracker: _ConcurrencyTracker,
    ) -> None:
        self._name = name
        self._execution_mode: ToolExecutionMode = execution_mode
        self._workspace = workspace
        self._tracker = tracker

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Deterministic benchmark file operation"

    @property
    def parameters(self) -> Mapping[str, object]:
        return {
            "type": "object",
            "properties": {
                "kind": {"enum": ["read", "write"]},
                "path": {"type": "string"},
                "delay_ms": {"type": "number", "minimum": 0},
                "content": {"type": "string"},
            },
            "required": ["kind", "path", "delay_ms", "content"],
            "additionalProperties": False,
        }

    @property
    def execution_mode(self) -> ToolExecutionMode:
        return self._execution_mode

    @property
    def examples(self) -> list[dict[str, object]]:
        return []

    async def execute(
        self,
        tool_call_id: str,
        params: ToolArguments,
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del signal, on_update
        kind = cast(OperationKind, params["kind"])
        path = self._resolve_path(str(params["path"]))
        delay_ms = cast(int | float, params["delay_ms"])
        self._tracker.enter(tool_call_id, kind)
        try:
            await asyncio.sleep(float(delay_ms) / 1_000)
            if kind == "read":
                content = path.read_text(encoding="utf-8")
                result = hashlib.sha256(content.encode()).hexdigest()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                content = str(params["content"])
                _append_text(path, content)
                result = hashlib.sha256(content.encode()).hexdigest()
            return AgentToolResult(content=[TextContent(text=result)])
        finally:
            self._tracker.leave(tool_call_id)

    def _resolve_path(self, value: str) -> Path:
        path = (self._workspace / _safe_relative_path(value)).resolve()
        _ensure_within(self._workspace, path, "tool path")
        return path


def _tool_call(operation: SchedulingOperation) -> ToolCallContent:
    name = "benchmark_read" if operation.kind == "read" else "benchmark_write"
    return ToolCallContent(
        id=operation.id,
        name=name,
        arguments={
            "kind": operation.kind,
            "path": operation.path,
            "delay_ms": operation.delay_ms,
            "content": operation.content,
        },
    )


def _append_text(path: Path, content: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(content)


def _ignore_event(event: AgentEvent) -> None:
    del event


def _load_operation(value: object, index: int) -> SchedulingOperation:
    data = _mapping(value, f"operations[{index}]")
    _reject_unknown(
        data,
        {"id", "kind", "path", "delay_ms", "content"},
        f"operations[{index}]",
    )
    operation_id = _text(data.get("id"), f"operations[{index}].id")
    if not _SAFE_TASK_ID.fullmatch(operation_id):
        raise ValueError(f"invalid operation id: {operation_id!r}")
    kind_value = _text(data.get("kind"), f"operations[{index}].kind")
    if kind_value not in {"read", "write"}:
        raise ValueError(f"unsupported operation kind: {kind_value!r}")
    delay = data.get("delay_ms", 0)
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
        raise ValueError(f"operations[{index}].delay_ms must be non-negative")
    return SchedulingOperation(
        id=operation_id,
        kind=cast(OperationKind, kind_value),
        path=_safe_relative_path(_text(data.get("path"), f"operations[{index}].path")),
        delay_ms=float(delay),
        content=_text(data.get("content", ""), f"operations[{index}].content"),
    )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_integer(value: object, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _reject_unknown(data: dict[str, object], allowed: set[str], field: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown fields in {field}: {', '.join(unknown)}")


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must stay relative: {value!r}")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError(f"path must stay relative: {value!r}")
    return normalized


def _ensure_within(root: Path, path: Path, field: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes workspace: {path}") from exc
