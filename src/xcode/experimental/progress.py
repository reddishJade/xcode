from __future__ import annotations

import logging
from typing import Any
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
