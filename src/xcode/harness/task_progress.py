"""任务进度断点和长任务租约状态。"""

from __future__ import annotations

import calendar
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from xcode.harness.skills import ToolInput, ToolSpec
from xcode.harness.task_store import CLAIMED, PENDING, TaskStore

logger = logging.getLogger("xcode.harness.task_progress")


@dataclass(frozen=True)
class TaskRunState:
    """长任务运行编排状态。"""

    task_id: int
    status: str
    attempt: int
    retry_limit: int
    lease_expires_at: str
    subtask_ids: list[int]


class TaskProgress:
    """管理长任务可重入进度断点与现场恢复的控制器。"""

    @staticmethod
    def save_progress(
        task_store: TaskStore, task_id: int | str, feature_list: list[dict[str, Any]]
    ) -> None:
        """原子性地更新 TaskStore 中的真值源（payload），并生成派生的只读 summary 视图文件。"""
        # 1. 更新真值源 (SoT) 并进行 filelock 原子性持久化
        task_store.update(task_id, payload={"feature_list": feature_list})

        # 2. 生成派生的人眼与模型快速读取只读视图
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

            lines.append(
                f"- [{status_char}] Step {idx}: {item.get('title', 'Untitled')}"
            )

        if in_progress_steps:
            lines.extend(["", "## Current Active Step:", f"- {in_progress_steps[0]}"])

        summary_content = "\n".join(lines) + "\n"

        # 3. 写入 claude-progress.txt 只读视图（位于 workspace root）
        progress_txt_path = task_store.root / "claude-progress.txt"
        progress_txt_path.write_text(summary_content, encoding="utf-8")

    @staticmethod
    def resume_task(task_store: TaskStore, task_id: int | str) -> list[dict[str, Any]]:
        """从 TaskStore 物理 JSON 文件（SoT）中精确读取并恢复特征列表现场。"""
        try:
            task = task_store.get(task_id)
            return task.payload.get("feature_list") or []
        except KeyError:
            logger.error("Failed to resume task: unknown task_id %s", task_id)
            return []

    @staticmethod
    def start_run(
        task_store: TaskStore,
        task_id: int | str,
        timeout_seconds: int = 3600,
        retry_limit: int = 0,
        subtasks: list[str] | None = None,
    ) -> TaskRunState:
        """启动或恢复一个带租约的长任务运行。"""
        task = task_store.get(task_id)
        orchestration = dict(task.payload.get("orchestration") or {})
        attempt = int(orchestration.get("attempt", 0)) + 1
        if attempt > retry_limit + 1:
            raise ValueError("retry limit exceeded")

        subtask_ids = _existing_subtask_ids(orchestration)
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
        payload = dict(task.payload)
        payload["orchestration"] = asdict(state)
        task_store.update(task.id, status=CLAIMED, payload=payload)
        return state

    @staticmethod
    def resume_run(task_store: TaskStore, task_id: int | str) -> TaskRunState:
        """恢复长任务运行状态。"""
        task = task_store.get(task_id)
        orchestration = dict(task.payload.get("orchestration") or {})
        return TaskRunState(
            task_id=task.id,
            status=str(orchestration.get("status", task.status)),
            attempt=int(orchestration.get("attempt", 0)),
            retry_limit=int(orchestration.get("retry_limit", 0)),
            lease_expires_at=str(orchestration.get("lease_expires_at", "")),
            subtask_ids=_existing_subtask_ids(orchestration),
        )

    @staticmethod
    def retry_run(task_store: TaskStore, task_id: int | str) -> TaskRunState:
        """在未超过 retry_limit 时重试长任务。"""
        task = task_store.get(task_id)
        orchestration = dict(task.payload.get("orchestration") or {})
        attempt = int(orchestration.get("attempt", 0))
        retry_limit = int(orchestration.get("retry_limit", 0))
        if attempt >= retry_limit + 1:
            raise ValueError("retry limit exceeded")
        return TaskProgress.start_run(
            task_store,
            task.id,
            timeout_seconds=_remaining_timeout(orchestration),
            retry_limit=retry_limit,
        )

    @staticmethod
    def expire_stale_runs(task_store: TaskStore) -> list[TaskRunState]:
        """将租约过期的运行标记为 timed_out 并释放为可重试任务。"""
        expired: list[TaskRunState] = []
        now = time.time()
        for task in task_store.list():
            orchestration = dict(task.payload.get("orchestration") or {})
            expires_at = _parse_timestamp(
                str(orchestration.get("lease_expires_at", ""))
            )
            if task.status != CLAIMED or expires_at <= 0 or expires_at >= now:
                continue
            orchestration["status"] = "timed_out"
            payload = dict(task.payload)
            payload["orchestration"] = orchestration
            updated = task_store.update(task.id, status=PENDING, payload=payload)
            expired.append(TaskProgress.resume_run(task_store, updated.id))
        return expired


def _existing_subtask_ids(orchestration: dict[str, Any]) -> list[int]:
    values = orchestration.get("subtask_ids") or []
    if not isinstance(values, list):
        return []
    return [int(value) for value in values if str(value).isdigit()]


def _future_timestamp(timeout_seconds: int) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() + max(1, timeout_seconds)),
    )


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return 0.0


def _remaining_timeout(orchestration: dict[str, Any]) -> int:
    expires_at = _parse_timestamp(str(orchestration.get("lease_expires_at", "")))
    if expires_at <= 0:
        return 3600
    return max(1, int(expires_at - time.time()))


def build_progress_tools(task_store: TaskStore) -> tuple[ToolSpec, ...]:
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
        TaskProgress.save_progress(task_store, task_id, checklist)
        return f"saved progress for task {task_id}"

    def resume_task_progress(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        checklist = TaskProgress.resume_task(task_store, task_id)
        return json.dumps(checklist, ensure_ascii=False, indent=2)

    def start_task_run(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        subtasks_raw = args.get("subtasks") or []
        if not isinstance(subtasks_raw, list):
            raise ValueError("subtasks must be an array")
        state = TaskProgress.start_run(
            task_store,
            task_id,
            timeout_seconds=int(args.get("timeout_seconds", 3600)),
            retry_limit=int(args.get("retry_limit", 0)),
            subtasks=[str(item) for item in subtasks_raw],
        )
        return json.dumps(asdict(state), ensure_ascii=False, indent=2)

    def resume_task_run(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        state = TaskProgress.resume_run(task_store, task_id)
        return json.dumps(asdict(state), ensure_ascii=False, indent=2)

    def retry_task_run(args: ToolInput) -> str:
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            raise ValueError("task_id is required")
        state = TaskProgress.retry_run(task_store, task_id)
        return json.dumps(asdict(state), ensure_ascii=False, indent=2)

    def expire_task_runs(_args: ToolInput) -> str:
        expired = TaskProgress.expire_stale_runs(task_store)
        return json.dumps(
            [asdict(state) for state in expired],
            ensure_ascii=False,
            indent=2,
        )

    return (
        ToolSpec(
            name="save_task_progress",
            description="Save a durable task checklist into TaskStore and write the read-only progress summary.",
            input_hint='{"task_id":1,"feature_list":[{"title":"Design","status":"completed"}]}',
            handler=save_task_progress,
            risk="low",
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
            risk="low",
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
            risk="low",
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
            risk="low",
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
            risk="low",
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
            risk="low",
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            group="progress",
        ),
    )
