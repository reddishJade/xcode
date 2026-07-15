"""Eval 领域契约测试。"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from xcode.evals.schema import (
    ArtifactManifest,
    ErrorCategory,
    ModelConfig,
    ResourceBudget,
    ResourceUsage,
    Task,
    TaskSource,
    Trial,
    TrialResult,
    Variant,
    VerifierResult,
    VerifierSpec,
)


def _budget() -> ResourceBudget:
    return ResourceBudget(wall_time_seconds=300, model_calls=20, tool_calls=100)


def _artifacts() -> ArtifactManifest:
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


def test_task_contains_only_opaque_verifier_reference() -> None:
    task = Task(
        task_id="xcode-fix-1",
        dataset_version="2026.07",
        prompt="修复真实回归。",
        source=TaskSource(
            kind="git_history",
            repository="xcode",
            revision="abc123",
            license="MIT",
        ),
        verifier_id="xcode-fix-1-hidden",
        allowed_paths=("src/xcode",),
        budget=_budget(),
    )

    payload = task.model_dump_json()

    assert "verifier_id" in payload
    assert "command" not in payload
    assert "hidden_root" not in payload


@pytest.mark.parametrize("path", ("/hidden/tests", "../hidden/tests"))
def test_task_rejects_paths_outside_workspace(path: str) -> None:
    with pytest.raises(ValidationError, match="workspace-relative"):
        Task(
            task_id="task-1",
            dataset_version="v1",
            prompt="修复问题。",
            source=TaskSource(
                kind="git_history",
                repository="xcode",
                revision="abc123",
                license="MIT",
            ),
            verifier_id="hidden-1",
            allowed_paths=(path,),
            budget=_budget(),
        )


def test_trial_records_full_comparable_configuration() -> None:
    trial = Trial(
        trial_id="experiment.task.full.0",
        experiment_id="experiment",
        task_id="task",
        dataset_version="v1",
        variant=Variant(
            variant_id="full",
            harness_revision="abc123",
            capabilities={"compaction": True},
        ),
        model=ModelConfig(provider="openai", model="model-id", seed=7),
        budget=_budget(),
        repetition=0,
        workspace_revision="parent123",
        command=("uv", "run", "xcode"),
    )

    assert trial.variant.capabilities == {"compaction": True}
    assert trial.model.seed == 7


def test_verifier_spec_requires_separate_absolute_hidden_root() -> None:
    with pytest.raises(ValidationError, match="hidden_root"):
        VerifierSpec(
            verifier_id="hidden-1",
            version="v1",
            command=("pytest", "hidden"),
            hidden_root="hidden",
            timeout_seconds=60,
        )


def test_success_requires_every_independent_check() -> None:
    now = datetime.now(UTC)
    verifier = VerifierResult(
        verifier_id="hidden-1",
        verifier_version="v1",
        completed=True,
        resolved=True,
        regression_free=True,
        policy_clean=True,
        log_artifact="verifier.log",
    )
    result = TrialResult(
        trial_id="trial-1",
        started_at=now,
        finished_at=now + timedelta(seconds=1),
        agent_completed=True,
        valid_trial=True,
        verifier=verifier,
        termination_reason="completed",
        usage=ResourceUsage(wall_time_seconds=1, model_calls=1, tool_calls=2),
        artifacts=_artifacts(),
    )

    assert result.success is True


def test_invalid_trial_requires_exclusion_category() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="error category"):
        TrialResult(
            trial_id="trial-1",
            started_at=now,
            finished_at=now,
            agent_completed=False,
            valid_trial=False,
            verifier=None,
            termination_reason="provider_error",
            usage=ResourceUsage(wall_time_seconds=1, model_calls=1, tool_calls=0),
            artifacts=_artifacts(),
        )


def test_infrastructure_failure_is_not_a_capability_failure() -> None:
    now = datetime.now(UTC)
    result = TrialResult(
        trial_id="trial-1",
        started_at=now,
        finished_at=now,
        agent_completed=False,
        valid_trial=False,
        verifier=None,
        error_category=ErrorCategory.PROVIDER_FAILURE,
        error_message="provider unavailable",
        termination_reason="provider_error",
        usage=ResourceUsage(wall_time_seconds=1, model_calls=1, tool_calls=0),
        artifacts=_artifacts(),
    )

    assert result.success is False
    assert result.valid_trial is False


def test_over_budget_trial_may_preserve_verifier_diagnostics() -> None:
    now = datetime.now(UTC)
    verifier = VerifierResult(
        verifier_id="hidden-1",
        verifier_version="v1",
        completed=True,
        resolved=False,
        regression_free=True,
        policy_clean=True,
        log_artifact="verifier.log",
    )
    result = TrialResult(
        trial_id="trial-1",
        started_at=now,
        finished_at=now,
        agent_completed=True,
        valid_trial=False,
        verifier=verifier,
        error_category=ErrorCategory.BUDGET_EXCEEDED,
        termination_reason="budget_exceeded",
        usage=ResourceUsage(wall_time_seconds=1, model_calls=2, tool_calls=3),
        artifacts=_artifacts(),
    )

    assert result.success is False
    assert result.verifier.regression_free is True
