"""Experiment artifact 的离线报告重建测试。"""

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from xcode.evals.artifacts import ArtifactStore
from xcode.evals.experiment import build_trials
from xcode.evals.reporting import ExperimentArtifactStore, ReportingError
from xcode.evals.schema import (
    ErrorCategory,
    Experiment,
    ModelConfig,
    ResourceBudget,
    ResourceUsage,
    Task,
    TaskSource,
    Trial,
    TrialResult,
    Variant,
    VerifierResult,
)


def _task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        dataset_version="v1",
        prompt="修复问题。",
        source=TaskSource(
            kind="git_history",
            repository="xcode",
            revision=f"parent-{task_id}",
            license="MIT",
        ),
        verifier_id=f"{task_id}-hidden",
        allowed_paths=("src",),
        budget=ResourceBudget(wall_time_seconds=60, model_calls=5, tool_calls=20),
    )


def _experiment() -> Experiment:
    return Experiment(
        experiment_id="offline-report",
        dataset_version="v1",
        task_ids=("task-a",),
        variants=(
            Variant(
                variant_id="full",
                harness_revision="abc123",
                capabilities={"full": True},
            ),
        ),
        model=ModelConfig(provider="provider", model="model"),
        repetitions=2,
        command=("xcode-eval",),
    )


def _persist_result(
    store: ArtifactStore,
    task: Task,
    trial: Trial,
    *,
    invalid: bool,
) -> None:
    paths = store.begin(task=task, trial=trial, environment={"python": "3.12"})
    paths.resolve(paths.manifest.trace).write_text("{}\n", encoding="utf-8")
    paths.resolve(paths.manifest.verifier_log).write_text("ok\n", encoding="utf-8")
    now = datetime.now(UTC)
    result = TrialResult(
        trial_id=trial.trial_id,
        started_at=now,
        finished_at=now,
        agent_completed=not invalid,
        valid_trial=not invalid,
        verifier=(
            None
            if invalid
            else VerifierResult(
                verifier_id=task.verifier_id,
                verifier_version="v1",
                completed=True,
                resolved=True,
                regression_free=True,
                policy_clean=True,
                log_artifact="verifier.log",
            )
        ),
        error_category=ErrorCategory.PROVIDER_FAILURE if invalid else None,
        termination_reason="provider_failure" if invalid else "completed",
        usage=ResourceUsage(
            wall_time_seconds=2,
            model_calls=1,
            tool_calls=3,
            input_tokens=100,
            output_tokens=10,
        ),
        artifacts=paths.manifest,
    )
    store.finish(paths=paths, patch="", result=result)


def test_report_is_rebuilt_only_from_verified_trial_artifacts(tmp_path: Path) -> None:
    experiment = _experiment()
    task = _task("task-a")
    control = ExperimentArtifactStore(tmp_path)
    root = control.begin(experiment)
    trials = build_trials(experiment, (task,))
    store = ArtifactStore(tmp_path)
    _persist_result(store, task, trials[0], invalid=False)
    _persist_result(store, task, trials[1], invalid=True)

    summary = control.rebuild(root)

    variant = summary.variants[0]
    assert variant.valid_trials == 1
    assert variant.excluded_trials == 1
    assert variant.success_rate == 1
    assert variant.usage.input_tokens == 200
    assert variant.usage.tokens_per_success == 200
    assert (root / "experiment.json").is_file()
    assert (root / "summary.json").is_file()
    assert (root / "report.md").is_file()
    assert (root / "experiment-checksums.json").is_file()
    lines = (root / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["artifact_root"] for line in lines)
    report = (root / "report.md").read_text(encoding="utf-8")
    assert "provider_failure=1" in report
    assert "including failures and exclusions" in report


def test_report_can_regenerate_tampered_derived_output(tmp_path: Path) -> None:
    experiment = _experiment()
    control = ExperimentArtifactStore(tmp_path)
    root = control.begin(experiment)
    control.rebuild(root)
    (root / "report.md").write_text("tampered", encoding="utf-8")

    control.rebuild(root)

    assert (root / "report.md").read_text(encoding="utf-8").startswith("# Experiment")


def test_report_rejects_tampered_experiment_declaration(tmp_path: Path) -> None:
    control = ExperimentArtifactStore(tmp_path)
    root = control.begin(_experiment())
    (root / "experiment.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReportingError, match="checksum"):
        control.rebuild(root)
