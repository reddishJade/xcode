"""任务进度断点和长任务租约状态。

orchestration state 由独立的 ``OrchestrationStore`` 管理，不再混入
``TaskStore.payload``。``save_progress`` 仅 merge ``feature_list``，
不会覆盖运行时控制面。所有函数显式要求 ``orchestration_store`` 参数。
"""

from __future__ import annotations

import calendar
import json
import logging
import time
from pathlib import Path
from typing import Any

from xcode.experimental.orchestration_store import OrchestrationStore, TaskRunState
from xcode.harness.skills import ToolInput, ToolSpec
from xcode.experimental.task_store import CLAIMED, PENDING, TaskStore

logger = logging.getLogger("xcode.experimental.task_progress")


def save_progress(
    task_store: TaskStore,
    task_id: int | str,
    feature_list: list[dict[str, Any]],
    *,
    summary_path: Path | None = None,
) -> None:
    """merge feature_list 进 task payload 并写只读 summary 文件。

    不再覆盖整个 payload——保留 blocked_by、parent_id 等其他字段。
    """
    current = task_store.get(task_id)
    new_payload = dict(current.payload)
    new_payload["feature_list"] = feature_list
    task_store.update(task_id, payload=new_payload)

    total = len(feature_list)
    completed = sum(1 for item in feature_list if item.get("status") == "completed")
    in_progress_steps = [
        item.get("title")
        for item in feature_list
        if item.get("status") == "in_progress"
    ]

    progress_percentage = (completed / total * 100.0) if total > 0 else 0.0

    lines = [
        "# Xcode Task Progress Summary (Read-Only View)",
        f"Task ID: {task_id}",
        f"Progress: {progress_percentage:.1f}% ({completed}/{total} steps completed)",
        "",
        "## Sub-task Checklist:",
    ]

    for idx, item in enumerate(feature_list, 1):
        status_char = " "
        status = item.get("status", "pending")
        if status == "completed":
            status_char = "x"
        elif status == "in_progress":
            status_char = "/"

        lines.append(f"- [{status_char}] Step {idx}: {item.get('title', 'Untitled')}")

    if in_progress_steps:
        lines.extend(["", "## Current Active Step:", f"- {in_progress_steps[0]}"])

    summary_content = "\n".join(lines) + "\n"
    target = summary_path or (task_store.root / ".local" / "progress_summary.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(summary_content, encoding="utf-8")


def resume_task(task_store: TaskStore, task_id: int | str) -> list[dict[str, Any]]:
    """从 TaskStore payload 读取 feature_list。"""
    try:
        task = task_store.get(task_id)
        return task.payload.get("feature_list") or []
    except KeyError:
        logger.error("Failed to resume task: unknown task_id %s", task_id)
        return []


def start_run(
    task_store: TaskStore,
    orchestration_store: OrchestrationStore,
    task_id: int | str,
    timeout_seconds: int = 3600,
    retry_limit: int = 0,
    subtasks: list[str] | None = None,
) -> TaskRunState:
    """启动或恢复一个带租约的长任务运行。

    orchestration 状态写入独立文件，不混入 task.payload。
    """
    task = task_store.get(task_id)
    prev_state = orchestration_store.get(task.id)
    attempt = (prev_state.attempt if prev_state else 0) + 1
    if attempt > retry_limit + 1:
        raise ValueError("retry limit exceeded")

    subtask_ids = list(prev_state.subtask_ids) if prev_state else []
    for title in subtasks or []:
        clean_title = title.strip()
        if not clean_title:
            continue
        child = task_store.create(
            clean_title,
            {"parent_id": task.id, "orchestration_role": "subtask"},
        )
        subtask_ids.append(child.id)

    state = TaskRunState(
        task_id=task.id,
        status="running",
        attempt=attempt,
        retry_limit=retry_limit,
        lease_expires_at=_future_timestamp(timeout_seconds),
        subtask_ids=subtask_ids,
    )
    orchestration_store.set(state)
    task_store.update(task.id, status=CLAIMED)
    return state


def resume_run(
    task_store: TaskStore,
    orchestration_store: OrchestrationStore,
    task_id: int | str,
) -> TaskRunState:
    """恢复长任务运行状态。

    独立文件缺失或字段不全时记录 warning 并返回 default state，
    不再静默用零值掩盖数据损坏。
    """
    tid = int(task_id)
    try:
        task = task_store.get(task_id)
    except KeyError:
        logger.warning("resume_run: unknown task %s", task_id)
        return _default_state(tid)

    state = orchestration_store.get(tid)
    if state is None:
        logger.warning("resume_run: no orchestration state for task %s", task_id)
        return _default_state(tid, fallback_status=task.status)

    missing: list[str] = []
    if not state.status:
        missing.append("status")
    if not state.lease_expires_at:
        missing.append("lease_expires_at")
    if missing:
        logger.warning(
            "resume_run: task %s orchestration missing fields %s", task_id, missing
        )
    return state


def retry_run(
    task_store: TaskStore,
    orchestration_store: OrchestrationStore,
    task_id: int | str,
) -> TaskRunState:
    """在未超过 retry_limit 时重试长任务。"""
    task = task_store.get(task_id)
    state = orchestration_store.get(task.id)
    if state is None:
        raise ValueError("no orchestration state; call start_run first")
    if state.attempt >= state.retry_limit + 1:
        raise ValueError("retry limit exceeded")
    return start_run(
        task_store,
        orchestration_store,
        task.id,
        timeout_seconds=_remaining_timeout_from_state(state),
        retry_limit=state.retry_limit,
    )


def expire_stale_runs(
    task_store: TaskStore,
    orchestration_store: OrchestrationStore,
) -> list[TaskRunState]:
    """将租约过期的运行标记为 timed_out 并释放为可重试任务。

    使用 lease 索引避免全表扫描 task 文件。
    """
    now = time.time()
    expired_task_ids = orchestration_store.list_expired(now)
    result: list[TaskRunState] = []
    for tid in expired_task_ids:
        state = orchestration_store.get(tid)
        if state is None:
            continue
        try:
            task = task_store.get(tid)
        except KeyError:
            orchestration_store.delete(tid)
            continue
        if task.status != CLAIMED:
            continue
        task_store.update(tid, status=PENDING)
        new_state = TaskRunState(
            task_id=state.task_id,
            status="timed_out",
            attempt=state.attempt,
            retry_limit=state.retry_limit,
            lease_expires_at=state.lease_expires_at,
            subtask_ids=list(state.subtask_ids),
        )
        orchestration_store.set(new_state)
        result.append(new_state)
    return result


def _default_state(task_id: int, *, fallback_status: str = "unknown") -> TaskRunState:
    return TaskRunState(
        task_id=task_id,
        status=fallback_status,
        attempt=0,
        retry_limit=0,
        lease_expires_at="",
        subtask_ids=[],
    )


def _future_timestamp(timeout_seconds: int) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() + max(1, timeout_seconds)),
    )


def _remaining_timeout_from_state(state: TaskRunState) -> int:
    expires_epoch = _parse_timestamp(state.lease_expires_at)
    if expires_epoch <= 0:
        return 3600
    return max(1, int(expires_epoch - time.time()))


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return 0.0


def build_progress_tools(
    task_store: TaskStore,
    orchestration_store: OrchestrationStore,
    *,
    summary_path: Path | None = None,
) -> tuple[ToolSpec, ...]:
    resolved_summary_path = summary_path

    def save_task_progress(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        feature_list = args.get("feature_list", args.get("checklist"))
        if task_id is None:
            raise ValueError("task_id is required")
        if not isinstance(feature_list, list):
            raise ValueError("feature_list must be an array")
        checklist: list[dict[str, Any]] = []
        for item in feature_list:
            if not isinstance(item, dict):
                raise ValueError("feature_list items must be objects")
            checklist.append(item)
        save_progress(
            task_store, task_id, checklist, summary_path=resolved_summary_path
        )
        return f"saved progress for task {task_id}"

    def resume_task_progress(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        checklist = resume_task(task_store, task_id)
        return json.dumps(checklist, ensure_ascii=False, indent=2)

    def start_task_run(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        subtasks_raw = args.get("subtasks") or []
        if not isinstance(subtasks_raw, list):
            raise ValueError("subtasks must be an array")
        state = start_run(
            task_store,
            orchestration_store,
            task_id,
            timeout_seconds=int(args.get("timeout_seconds", 3600)),
            retry_limit=int(args.get("retry_limit", 0)),
            subtasks=[str(item) for item in subtasks_raw],
        )
        return json.dumps(_state_to_dict(state), ensure_ascii=False, indent=2)

    def resume_task_run(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        state = resume_run(task_store, orchestration_store, task_id)
        return json.dumps(_state_to_dict(state), ensure_ascii=False, indent=2)

    def retry_task_run(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        state = retry_run(task_store, orchestration_store, task_id)
        return json.dumps(_state_to_dict(state), ensure_ascii=False, indent=2)

    def expire_task_runs(_args: ToolInput) -> str:
        expired = expire_stale_runs(task_store, orchestration_store)
        return json.dumps(
            [_state_to_dict(state) for state in expired],
            ensure_ascii=False,
            indent=2,
        )

    return (
        ToolSpec(
            name="save_task_progress",
            description="Save a durable task checklist into TaskStore and write the read-only progress summary.",
            input_hint='{"task_id":1,"feature_list":[{"title":"Design","status":"completed"}]}',
            handler=save_task_progress,
            schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "feature_list": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["task_id", "feature_list"],
                "additionalProperties": False,
            },
            group="progress",
        ),
        ToolSpec(
            name="resume_task_progress",
            description="Load the durable task checklist from TaskStore.",
            input_hint='{"task_id":1}',
            handler=resume_task_progress,
            schema={
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            read_only=True,
            group="progress",
        ),
        ToolSpec(
            name="start_task_run",
            description=(
                "Start or resume a leased long-running task and optionally "
                "dispatch subtasks."
            ),
            input_hint=(
                '{"task_id":1,"timeout_seconds":3600,'
                '"retry_limit":1,"subtasks":["Write tests"]}'
            ),
            handler=start_task_run,
            schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "timeout_seconds": {"type": "integer"},
                    "retry_limit": {"type": "integer"},
                    "subtasks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            group="progress",
        ),
        ToolSpec(
            name="resume_task_run",
            description="Read the current orchestration state for a long-running task.",
            input_hint='{"task_id":1}',
            handler=resume_task_run,
            schema={
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            read_only=True,
            group="progress",
        ),
        ToolSpec(
            name="retry_task_run",
            description=(
                "Retry a timed-out or failed long-running task if retry budget remains."
            ),
            input_hint='{"task_id":1}',
            handler=retry_task_run,
            schema={
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            group="progress",
        ),
        ToolSpec(
            name="expire_task_runs",
            description="Release long-running tasks whose orchestration lease expired.",
            input_hint="{}",
            handler=expire_task_runs,
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            group="progress",
        ),
    )


def _state_to_dict(state: TaskRunState) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(state)
