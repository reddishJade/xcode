"""Xcode Eval 的真实 Experiment 命令入口。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from xcode.harness.config import discover_runtime_config

from .artifacts import ArtifactStore
from .dataset import load_tasks
from .experiment import build_trials
from .isolation import BubblewrapExecutor
from .policy import build_eval_runtime, EVAL_EXECUTION_MODE
from .reporting import ExperimentArtifactStore
from .schema import Experiment, ModelConfig, Task, Variant, VerifierSpec
from .trial_runner import TrialRunner


def main(argv: list[str] | None = None) -> int:
    """运行或恢复真实 Experiment；Agent 自报测试不作为能力分。"""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.all_tasks and not args.task:
        parser.error("select at least one --task or use --all-tasks")
    if args.all_tasks and args.task:
        parser.error("--all-tasks cannot be combined with --task")
    if args.repetition < 0:
        parser.error("--repetition must be non-negative")
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    control_root = Path.cwd().resolve()
    dataset_tasks = load_tasks(args.dataset)
    tasks_by_id = {task.task_id: task for task in dataset_tasks}
    task_ids = tuple(tasks_by_id) if args.all_tasks else tuple(args.task)
    try:
        selected_tasks = tuple(tasks_by_id[task_id] for task_id in task_ids)
    except KeyError as error:
        parser.error(f"unknown task: {error.args[0]}")
    dataset_versions = {task.dataset_version for task in selected_tasks}
    if len(dataset_versions) != 1:
        parser.error("selected tasks must use one dataset version")

    runtime = build_eval_runtime(discover_runtime_config(control_root, args.config))
    profile = runtime.provider.model_profiles["main"]
    harness_revision = _git(control_root, "rev-parse", "HEAD")
    dirty = bool(_git(control_root, "status", "--porcelain"))
    sanitized_runtime = runtime.model_dump(mode="json")
    for model_profile in sanitized_runtime["provider"]["model_profiles"].values():
        model_profile["api_key"] = "<redacted-configured>"
    variant = _full_variant(
        harness_revision=harness_revision,
        sanitized_runtime=sanitized_runtime,
    )
    repetitions = (
        args.repetitions if args.repetitions is not None else args.repetition + 1
    )
    command = tuple(sys.argv if argv is None else ("xcode-eval", *argv))
    experiment = Experiment(
        experiment_id=args.experiment_id,
        dataset_version=next(iter(dataset_versions)),
        task_ids=task_ids,
        variants=(variant,),
        model=ModelConfig(
            provider=profile.transport,
            model=profile.chat_model,
            options={
                "base_url": profile.base_url,
                "thinking": profile.thinking,
                "reasoning_effort": profile.reasoning_effort,
            },
        ),
        repetitions=repetitions,
        command=command,
    )
    scheduled = build_trials(experiment, selected_tasks)
    if args.repetitions is None:
        scheduled = tuple(
            trial for trial in scheduled if trial.repetition == args.repetition
        )

    python_runtime = Path(sys.executable).resolve().parents[1]
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv executable is required for isolated Eval")
    executor = BubblewrapExecutor(
        xcode_source=control_root / "src/xcode",
        virtualenv=control_root / ".venv",
        python_runtime=python_runtime,
        uv_executable=Path(uv_executable),
    )
    artifact_root = args.run_root / "artifacts"
    experiment_store = ExperimentArtifactStore(artifact_root)
    experiment_root = experiment_store.begin(experiment)
    completed = {
        record.trial.trial_id
        for record in experiment_store.load_records(experiment_root)
    }
    runner = TrialRunner(
        repository=args.repository,
        workspace_root=args.run_root / "workspaces",
        artifact_store=ArtifactStore(artifact_root),
        executor=executor,
    )
    for trial in scheduled:
        if trial.trial_id in completed:
            continue
        _quarantine_incomplete(
            experiment_root / trial.trial_id,
            reason="artifact",
        )
        _quarantine_incomplete(
            args.run_root / "workspaces" / trial.trial_id,
            reason="workspace",
        )
        task = tasks_by_id[trial.task_id]
        result = runner.run(
            task=task,
            trial=trial,
            verifier_spec=_verifier_spec(
                control_root=control_root,
                private_root=args.private_root,
                task=task,
            ),
            runtime_config=runtime,
            environment={
                "captured_at": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "xcode_revision": harness_revision,
                "xcode_dirty": dirty,
                "workspace_revision": task.source.revision,
                "dataset_version": task.dataset_version,
                "provider": profile.transport,
                "model": profile.chat_model,
                "base_url": profile.base_url,
                "provider_auth_configured": bool(profile.api_key),
            },
        )
        completed.add(result.trial_id)
        experiment_store.rebuild(experiment_root)

    summary = experiment_store.rebuild(experiment_root)
    invalid = sum(variant.excluded_trials for variant in summary.variants)
    print(
        json.dumps(
            {
                "experiment_id": experiment.experiment_id,
                "scheduled_trials": len(scheduled),
                "observed_trials": len(summary.trials),
                "invalid_trials": invalid,
                "artifact_root": str(experiment_root),
                "summary": str(experiment_root / "summary.json"),
                "report": str(experiment_root / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if invalid == 0 else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xcode-eval")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--repetitions", type=int)
    return parser


def _full_variant(
    *,
    harness_revision: str,
    sanitized_runtime: dict[str, Any],
) -> Variant:
    return Variant(
        variant_id="full",
        harness_revision=harness_revision,
        capabilities={
            "context_assembly": True,
            "tools": True,
            "compaction": True,
            "error_recovery": True,
            "permission_feedback": True,
            "session": True,
            "mcp": True,
            "memory": True,
        },
        config={
            "runtime": sanitized_runtime,
            "execution_mode": EVAL_EXECUTION_MODE,
            "isolation": "bubblewrap",
            "external_network_tools": "denied",
        },
    )


def _verifier_spec(
    *,
    control_root: Path,
    private_root: Path,
    task: Task,
) -> VerifierSpec:
    return VerifierSpec(
        verifier_id=task.verifier_id,
        version="v1",
        command=(str(control_root / ".venv/bin/python"), "verify.py", "{workspace}"),
        hidden_root=str((private_root / task.task_id).resolve()),
        timeout_seconds=180,
    )


def _quarantine_incomplete(path: Path, *, reason: str) -> None:
    """保留中断现场，同时确保重跑使用全新目录。"""
    if not path.exists():
        return
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path.replace(path.with_name(f"{path.name}.incomplete-{reason}-{suffix}"))


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
