"""SWE-bench Lite 小样本的任务转换和独立评分桥接。

本模块只处理公开题面、Agent 产出的 patch 和外部评分命令。它不加载
``test_patch``、``FAIL_TO_PASS`` 等隐藏材料，也不把外部评分器放进 Agent
工作区。prediction 生成和外部评分始终是两个独立阶段。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import ValidationError

from .artifacts import ArtifactError, ArtifactStore
from .schema import ResourceBudget, Task, TaskSource, Trial, TrialResult

SWE_BENCH_LITE_KIND = "swe-bench-lite"
SWE_BENCH_LITE_FAST_VERIFIER_ID = "swebench-lite-fast"


class SWEbenchLiteError(RuntimeError):
    """公开任务、prediction 或外部评分输入不符合契约。"""


@dataclass(frozen=True)
class SWEbenchLiteInstance:
    """仅保留 Agent 可以看到的 SWE-bench Lite 字段。"""

    instance_id: str
    repository: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SWEbenchLiteInstance":
        """从上游 JSON 行选择公开字段，忽略隐藏测试和标准答案字段。"""
        try:
            instance_id = _required_text(value, "instance_id")
            repository = _required_text(value, "repo")
            base_commit = _required_text(value, "base_commit")
            problem_statement = _required_text(value, "problem_statement")
        except (KeyError, TypeError, ValueError) as error:
            raise SWEbenchLiteError(
                f"invalid SWE-bench Lite instance: {error}"
            ) from error
        return cls(
            instance_id=instance_id,
            repository=repository,
            base_commit=base_commit,
            problem_statement=problem_statement,
        )


@dataclass(frozen=True)
class Prediction:
    """SWE-bench 评分器认可的三字段 prediction。"""

    instance_id: str
    model_name_or_path: str
    model_patch: str

    def as_dict(self) -> dict[str, str]:
        """按官方 prediction JSONL 字段名导出。"""
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


def load_instances_jsonl(path: Path) -> tuple[SWEbenchLiteInstance, ...]:
    """加载公开数据快照，拒绝空文件和重复 instance id。"""
    instances: list[SWEbenchLiteInstance] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SWEbenchLiteError(f"cannot read instances: {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as error:
            raise SWEbenchLiteError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise SWEbenchLiteError(
                f"instance at {path}:{line_number} is not an object"
            )
        instances.append(SWEbenchLiteInstance.from_mapping(value))
    if not instances:
        raise SWEbenchLiteError(f"SWE-bench Lite data contains no instances: {path}")
    identifiers = [instance.instance_id for instance in instances]
    if len(set(identifiers)) != len(identifiers):
        raise SWEbenchLiteError("SWE-bench Lite data contains duplicate instance ids")
    return tuple(instances)


def to_task(
    instance: SWEbenchLiteInstance,
    *,
    dataset_version: str,
    license_name: str,
    budget: ResourceBudget,
) -> Task:
    """把公开实例转为统一 Task，保持官方评分在控制面之外。"""
    return Task(
        task_id=instance.instance_id,
        dataset_version=dataset_version,
        prompt=(
            "You are solving a SWE-bench Lite issue in the checked-out repository.\n"
            f"Instance: {instance.instance_id}\n"
            f"Base commit: {instance.base_commit}\n\n"
            "Problem statement:\n"
            f"{instance.problem_statement.strip()}\n\n"
            "Edit the repository to resolve the issue. Use local validation when useful; "
            "do not fetch network resources or modify benchmark infrastructure."
        ),
        source=TaskSource(
            kind=SWE_BENCH_LITE_KIND,
            repository=f"https://github.com/{instance.repository}.git",
            revision=instance.base_commit,
            license=license_name,
            upstream_id=instance.instance_id,
        ),
        verifier_id=SWE_BENCH_LITE_FAST_VERIFIER_ID,
        allowed_paths=(".",),
        tags=("external", "swebench-lite"),
        budget=budget,
        known_limitations=(
            "Official scoring is a separate control-plane step; prediction generation "
            "alone is not a benchmark result.",
        ),
    )


def export_predictions(
    artifact_roots: Iterable[Path],
    *,
    output: Path,
    model_name_or_path: str | None = None,
) -> tuple[Prediction, ...]:
    """从已封存 Trial 导出 prediction，不重跑 Agent 或改写原 artifact。"""
    predictions: list[Prediction] = []
    seen: set[str] = set()
    for root in artifact_roots:
        task, trial, result = _load_trial_artifact(root)
        if task.source.kind != SWE_BENCH_LITE_KIND:
            raise SWEbenchLiteError(f"artifact is not a SWE-bench Lite task: {root}")
        if not result.agent_completed:
            raise SWEbenchLiteError(
                f"Agent did not complete; cannot export prediction: {root}"
            )
        instance_id = task.source.upstream_id or task.task_id
        if instance_id in seen:
            raise SWEbenchLiteError(
                f"multiple predictions for one instance are ambiguous: {instance_id}"
            )
        seen.add(instance_id)
        try:
            patch = (root / result.artifacts.patch).read_text(encoding="utf-8")
        except OSError as error:
            raise SWEbenchLiteError(
                f"cannot read patch from {root}: {error}"
            ) from error
        predictions.append(
            Prediction(
                instance_id=instance_id,
                model_name_or_path=model_name_or_path or trial.model.model,
                model_patch=patch,
            )
        )
    if not predictions:
        raise SWEbenchLiteError(
            "no Trial artifacts were selected for prediction export"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(prediction.as_dict(), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for prediction in predictions
        ),
        encoding="utf-8",
    )
    return tuple(predictions)


def build_fast_command(
    *,
    binary: Path,
    dataset: Path,
    predictions: Path,
    run_id: str,
    output: Path,
    workers: int = 1,
    timeout_seconds: int = 900,
) -> tuple[str, ...]:
    """构造 swe-bench-fast 的独立评分命令。"""
    if workers < 1:
        raise SWEbenchLiteError("workers must be at least 1")
    if timeout_seconds < 1:
        raise SWEbenchLiteError("timeout_seconds must be at least 1")
    return (
        str(binary),
        "run",
        "--dataset",
        str(dataset),
        "--predictions",
        str(predictions),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout_seconds),
        "--run-id",
        run_id,
        "--output",
        str(output),
    )


def _load_trial_artifact(root: Path) -> tuple[Task, Trial, TrialResult]:
    root = root.resolve()
    try:
        task = Task.model_validate_json(
            (root / "task.json").read_text(encoding="utf-8")
        )
        trial = Trial.model_validate_json(
            (root / "trial.json").read_text(encoding="utf-8")
        )
        result = ArtifactStore(root.parents[1]).load_result(root, verify=True)
    except (OSError, ValueError, ValidationError, ArtifactError) as error:
        raise SWEbenchLiteError(f"invalid Trial artifact {root}: {error}") from error
    return task, trial, result


def _required_text(value: Mapping[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """提供小型命令入口，避免手写 prediction 或评分命令。"""
    parser = argparse.ArgumentParser(prog="xcode-eval-swebench-lite")
    subparsers = parser.add_subparsers(dest="action", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument(
        "--artifact-root", action="append", type=Path, required=True
    )
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--model-name")
    command_parser = subparsers.add_parser("fast-command")
    command_parser.add_argument("--binary", type=Path, required=True)
    command_parser.add_argument("--dataset", type=Path, required=True)
    command_parser.add_argument("--predictions", type=Path, required=True)
    command_parser.add_argument("--run-id", required=True)
    command_parser.add_argument("--output", type=Path, required=True)
    command_parser.add_argument("--workers", type=int, default=1)
    command_parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.action == "export":
        predictions = export_predictions(
            args.artifact_root,
            output=args.output,
            model_name_or_path=args.model_name,
        )
        print(json.dumps({"predictions": len(predictions), "output": str(args.output)}))
        return 0
    command = build_fast_command(
        binary=args.binary,
        dataset=args.dataset,
        predictions=args.predictions,
        run_id=args.run_id,
        output=args.output,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
