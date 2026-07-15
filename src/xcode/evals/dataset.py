"""版本化 Task 数据集的加载与一致性检查。"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .schema import Task


class DatasetError(RuntimeError):
    """数据集结构、版本或 Task 内容无效。"""


def load_tasks(dataset_root: Path) -> tuple[Task, ...]:
    """加载 JSON Task，并拒绝重复标识或混合版本。"""
    paths = sorted((dataset_root / "tasks").glob("*.json"))
    if not paths:
        raise DatasetError(f"dataset contains no tasks: {dataset_root}")
    tasks: list[Task] = []
    for path in paths:
        try:
            tasks.append(Task.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValidationError, ValueError) as error:
            raise DatasetError(f"invalid task {path}: {error}") from error
    identifiers = [task.task_id for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetError("dataset contains duplicate task ids")
    versions = {task.dataset_version for task in tasks}
    if len(versions) != 1:
        raise DatasetError(f"dataset mixes versions: {sorted(versions)}")
    return tuple(tasks)
