"""Experiment 声明到严格配对 Trial 的确定性展开。"""

from __future__ import annotations

from collections.abc import Iterable

from .schema import Experiment, Task, Trial


class ExperimentError(RuntimeError):
    """Experiment 与数据集不一致或不能形成配对 Trial。"""


def build_trials(experiment: Experiment, tasks: Iterable[Task]) -> tuple[Trial, ...]:
    """按 repetition、task、variant 顺序建立不共享状态的 Trial。"""
    by_id = {task.task_id: task for task in tasks}
    selected: list[Task] = []
    for task_id in experiment.task_ids:
        task = by_id.get(task_id)
        if task is None:
            raise ExperimentError(f"experiment task is missing: {task_id}")
        if task.dataset_version != experiment.dataset_version:
            raise ExperimentError(
                f"task {task_id!r} belongs to dataset {task.dataset_version!r}, "
                f"not {experiment.dataset_version!r}"
            )
        selected.append(task)

    trials: list[Trial] = []
    for repetition in range(experiment.repetitions):
        for task in selected:
            for variant in experiment.variants:
                trials.append(
                    Trial(
                        trial_id=(
                            f"{experiment.experiment_id}.{task.task_id}."
                            f"{variant.variant_id}.r{repetition}"
                        ),
                        experiment_id=experiment.experiment_id,
                        task_id=task.task_id,
                        dataset_version=experiment.dataset_version,
                        variant=variant,
                        model=experiment.model,
                        budget=task.budget,
                        repetition=repetition,
                        workspace_revision=task.source.revision,
                        command=experiment.command,
                    )
                )
    return tuple(trials)
