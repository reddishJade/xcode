"""Experiment 配对展开与聚合公式测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from xcode.evals.experiment import build_trials, ExperimentError
from xcode.evals.metrics import aggregate_experiment, TrialRecord
from xcode.evals.schema import (
    ArtifactManifest,
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


def _budget() -> ResourceBudget:
    return ResourceBudget(wall_time_seconds=60, model_calls=5, tool_calls=20)


def _task(task_id: str, *, dataset: str = "v1") -> Task:
    return Task(
        task_id=task_id,
        dataset_version=dataset,
        prompt="修复真实问题。",
        source=TaskSource(
            kind="git_history",
            repository="xcode",
            revision=f"parent-{task_id}",
            license="MIT",
        ),
        verifier_id=f"{task_id}-hidden",
        allowed_paths=("src",),
        budget=_budget(),
    )


def _variant(variant_id: str) -> Variant:
    return Variant(
        variant_id=variant_id,
        harness_revision="abc123",
        capabilities={"full": variant_id == "full"},
    )


def _experiment() -> Experiment:
    return Experiment(
        experiment_id="baseline-1",
        dataset_version="v1",
        task_ids=("task-a", "task-b"),
        variants=(_variant("full"), _variant("minimal")),
        model=ModelConfig(provider="deepseek_chat", model="model-id"),
        repetitions=2,
        command=("uv", "run", "xcode-eval"),
    )


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        task="task.json",
        trial="trial.json",
        trace="trace.jsonl",
        patch="changes.patch",
        stdout="agent.stdout",
        stderr="agent.stderr",
        verifier_log="verifier.log",
        result="result.json",
        environment="environment.json",
    )


def _result(
    trial: Trial,
    *,
    success: bool,
    input_tokens: int,
    invalid: ErrorCategory | None = None,
) -> TrialResult:
    now = datetime.now(UTC)
    verifier = (
        None
        if invalid is not None
        else VerifierResult(
            verifier_id=f"{trial.task_id}-hidden",
            verifier_version="v1",
            completed=True,
            resolved=success,
            regression_free=True,
            policy_clean=True,
            log_artifact="verifier.log",
        )
    )
    return TrialResult(
        trial_id=trial.trial_id,
        started_at=now,
        finished_at=now,
        agent_completed=invalid is None,
        valid_trial=invalid is None,
        verifier=verifier,
        error_category=invalid,
        termination_reason=invalid.value if invalid is not None else "completed",
        usage=ResourceUsage(
            wall_time_seconds=input_tokens / 10,
            model_calls=1,
            tool_calls=2,
            input_tokens=input_tokens,
            output_tokens=5,
        ),
        artifacts=_manifest(),
    )


def test_experiment_builds_strictly_paired_unique_trials() -> None:
    trials = build_trials(_experiment(), (_task("task-a"), _task("task-b")))

    assert len(trials) == 8
    assert [trial.trial_id for trial in trials[:4]] == [
        "baseline-1.task-a.full.r0",
        "baseline-1.task-a.minimal.r0",
        "baseline-1.task-b.full.r0",
        "baseline-1.task-b.minimal.r0",
    ]
    assert trials[4].trial_id == "baseline-1.task-a.full.r1"
    assert trials[0].budget == _task("task-a").budget
    assert trials[0].workspace_revision == "parent-task-a"


def test_experiment_rejects_missing_or_foreign_dataset_tasks() -> None:
    with pytest.raises(ExperimentError, match="missing"):
        build_trials(_experiment(), (_task("task-a"),))

    with pytest.raises(ExperimentError, match="belongs to dataset"):
        build_trials(
            _experiment(),
            (_task("task-a"), _task("task-b", dataset="v2")),
        )


def test_experiment_rejects_duplicate_comparison_axes() -> None:
    payload = _experiment().model_dump()
    payload["task_ids"] = ("task-a", "task-a")
    with pytest.raises(ValidationError, match="task_ids must be unique"):
        Experiment.model_validate(payload)


def test_aggregate_excludes_invalid_results_but_keeps_their_cost() -> None:
    experiment = _experiment()
    full_trials = [
        trial
        for trial in build_trials(experiment, (_task("task-a"), _task("task-b")))
        if trial.variant.variant_id == "full"
    ]
    records = (
        TrialRecord(
            full_trials[0], _result(full_trials[0], success=True, input_tokens=10)
        ),
        TrialRecord(
            full_trials[1], _result(full_trials[1], success=True, input_tokens=20)
        ),
        TrialRecord(
            full_trials[2], _result(full_trials[2], success=False, input_tokens=30)
        ),
        TrialRecord(
            full_trials[3],
            _result(
                full_trials[3],
                success=False,
                input_tokens=40,
                invalid=ErrorCategory.PROVIDER_FAILURE,
            ),
        ),
    )

    summary = aggregate_experiment(experiment, records)
    full, minimal = summary.variants

    assert full.valid_trials == 3
    assert full.excluded_trials == 1
    assert full.successes == 2
    assert full.success_rate == pytest.approx(2 / 3)
    assert full.pass_k == 2
    assert full.pass_at_k == 1
    assert full.pass_power_k == 0
    assert full.pass_k_eligible_tasks == 1
    assert full.exclusions == {ErrorCategory.PROVIDER_FAILURE: 1}
    assert full.usage.input_tokens == 100
    assert full.usage.tokens_per_success == 50
    assert full.usage.tool_calls == 8
    assert minimal.observed_trials == 0
    assert minimal.missing_trials == 4
    assert summary.efficient_variant_ids == ("full",)
    assert len(summary.trials) == 4
    assert summary.comparisons[0].observed_pairs == 0


def test_aggregate_computes_only_strictly_paired_harness_gain() -> None:
    experiment = _experiment()
    trials = build_trials(experiment, (_task("task-a"), _task("task-b")))
    selected = {
        (trial.task_id, trial.variant.variant_id): trial
        for trial in trials
        if trial.repetition == 0
    }
    outcomes = {
        ("task-a", "full"): (True, 20),
        ("task-a", "minimal"): (False, 10),
        ("task-b", "full"): (False, 20),
        ("task-b", "minimal"): (True, 10),
    }
    records = tuple(
        TrialRecord(
            selected[key],
            _result(
                selected[key],
                success=success,
                input_tokens=input_tokens,
            ),
        )
        for key, (success, input_tokens) in outcomes.items()
    )

    comparison = aggregate_experiment(experiment, records).comparisons[0]

    assert comparison.candidate_variant_id == "full"
    assert comparison.control_variant_id == "minimal"
    assert comparison.declared_pairs == 4
    assert comparison.observed_pairs == 2
    assert comparison.missing_pairs == 2
    assert comparison.valid_pairs == 2
    assert comparison.candidate_wins == 1
    assert comparison.control_wins == 1
    assert comparison.ties == 0
    assert comparison.harness_gain == 0
    assert comparison.input_tokens_delta == 20


def test_aggregate_rejects_duplicate_trial_artifacts() -> None:
    experiment = _experiment()
    trial = build_trials(experiment, (_task("task-a"), _task("task-b")))[0]
    record = TrialRecord(trial, _result(trial, success=True, input_tokens=10))

    with pytest.raises(ValueError, match="duplicate trial artifact"):
        aggregate_experiment(experiment, (record, record))
