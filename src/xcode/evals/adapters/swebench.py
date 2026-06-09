from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xcode.evals.parameters import EvalReport, EvalTask, TrialResult


def build_swebench_predictions(
    report: EvalReport,
    tasks: tuple[EvalTask, ...],
    *,
    model_name: str,
) -> tuple[dict[str, Any], ...]:
    """构造 SWE-bench predictions JSONL 记录。"""
    task_by_id = {task.id: task for task in tasks}
    predictions: list[dict[str, Any]] = []
    for task_id in sorted(task_by_id):
        task = task_by_id[task_id]
        instance_id = _swebench_instance_id(task)
        if not instance_id:
            continue
        trial = _select_prediction_trial(
            tuple(t for t in report.trials if t.task_id == task_id)
        )
        model_patch = trial.metrics.get("model_patch", "") if trial else ""
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": str(model_patch),
            }
        )
    return tuple(predictions)


def write_swebench_predictions(
    report: EvalReport,
    tasks: tuple[EvalTask, ...],
    path: Path,
    *,
    model_name: str,
) -> Path:
    """写入 SWE-bench predictions JSONL 文件。"""
    predictions = build_swebench_predictions(report, tasks, model_name=model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in predictions]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _swebench_instance_id(task: EvalTask) -> str:
    benchmark = task.metadata.get("benchmark", {})
    if not isinstance(benchmark, dict):
        return ""
    name = str(benchmark.get("name", ""))
    if name not in {"swebench-lite", "swebench-verified"}:
        return ""
    return str(benchmark.get("instance_id", "")).strip()


def _select_prediction_trial(trials: tuple[TrialResult, ...]) -> TrialResult | None:
    for trial in trials:
        if trial.success and trial.metrics.get("model_patch"):
            return trial
    for trial in trials:
        if trial.metrics.get("model_patch"):
            return trial
    return trials[0] if trials else None
