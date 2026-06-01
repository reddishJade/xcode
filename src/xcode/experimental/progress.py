from __future__ import annotations

import logging
import json
from typing import Any

from ..harness.skills import ToolSpec
from .tasks import TaskStore

logger = logging.getLogger("xcode.experimental.progress")


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


def build_progress_tools(task_store: TaskStore) -> tuple[ToolSpec, ...]:
    from ..harness.skills import parse_tool_input

    def save_task_progress(action_input: str) -> str:
        args = parse_tool_input(action_input)
        task_id = args.get("task_id", args.get("id"))
        feature_list = args.get("feature_list", args.get("checklist"))
        if task_id is None:
            return "task_id is required"
        if not isinstance(feature_list, list):
            return "feature_list must be an array"
        checklist: list[dict[str, Any]] = []
        for item in feature_list:
            if not isinstance(item, dict):
                return "feature_list items must be objects"
            checklist.append(item)
        TaskProgress.save_progress(task_store, task_id, checklist)
        return f"saved progress for task {task_id}"

    def resume_task_progress(action_input: str) -> str:
        args = parse_tool_input(action_input)
        task_id = args.get("task_id", args.get("id"))
        if task_id is None:
            return "task_id is required"
        checklist = TaskProgress.resume_task(task_store, task_id)
        return json.dumps(checklist, ensure_ascii=False, indent=2)

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
    )
