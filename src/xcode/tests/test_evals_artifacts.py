"""Trial artifact 的完整性和离线重建测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xcode.evals.artifacts import ArtifactError, ArtifactStore
from xcode.evals.schema import (
    ErrorCategory,
    ModelConfig,
    ResourceBudget,
    ResourceUsage,
    Task,
    TaskSource,
    Trial,
    TrialResult,
    Variant,
)


def _task() -> Task:
    return Task(
        task_id="task-1",
        dataset_version="v1",
        prompt="修复问题。",
        source=TaskSource(
            kind="git_history",
            repository="xcode",
            revision="parent",
            license="MIT",
        ),
        verifier_id="hidden-1",
        allowed_paths=("src",),
        budget=ResourceBudget(wall_time_seconds=60, model_calls=5, tool_calls=20),
    )


def _trial() -> Trial:
    return Trial(
        trial_id="trial-1",
        experiment_id="experiment-1",
        task_id="task-1",
        dataset_version="v1",
        variant=Variant(
            variant_id="full",
            harness_revision="abc123",
            capabilities={"full": True},
        ),
        model=ModelConfig(provider="provider", model="model"),
        budget=_task().budget,
        repetition=0,
        workspace_revision="parent",
        command=("xcode-eval", "run"),
    )


def test_artifacts_can_rebuild_result_offline(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    paths = store.begin(
        task=_task(),
        trial=_trial(),
        environment={"python": "3.12", "platform": "test"},
    )
    paths.resolve(paths.manifest.trace).write_text("{}\n", encoding="utf-8")
    paths.resolve(paths.manifest.verifier_log).write_text("failed\n", encoding="utf-8")
    now = datetime.now(UTC)
    result = TrialResult(
        trial_id="trial-1",
        started_at=now,
        finished_at=now,
        agent_completed=True,
        valid_trial=False,
        verifier=None,
        error_category=ErrorCategory.VERIFIER_FAILURE,
        termination_reason="verifier_failure",
        usage=ResourceUsage(wall_time_seconds=1, model_calls=1, tool_calls=2),
        artifacts=paths.manifest,
    )
    store.finish(paths=paths, patch="diff", result=result)

    rebuilt = store.load_result(paths.root)

    assert rebuilt == result
    for relative in paths.manifest.model_dump().values():
        assert paths.resolve(relative).is_file()


def test_artifact_integrity_detects_tampering(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    paths = store.begin(
        task=_task(),
        trial=_trial(),
        environment={"python": "3.12"},
    )
    paths.resolve(paths.manifest.trace).write_text("{}\n", encoding="utf-8")
    paths.resolve(paths.manifest.verifier_log).write_text("ok\n", encoding="utf-8")
    now = datetime.now(UTC)
    result = TrialResult(
        trial_id="trial-1",
        started_at=now,
        finished_at=now,
        agent_completed=True,
        valid_trial=False,
        verifier=None,
        error_category=ErrorCategory.AGENT_FAILURE,
        termination_reason="agent_failure",
        usage=ResourceUsage(wall_time_seconds=1, model_calls=1, tool_calls=1),
        artifacts=paths.manifest,
    )
    store.finish(paths=paths, patch="diff", result=result)
    paths.resolve(paths.manifest.patch).write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactError, match="checksum"):
        store.load_result(paths.root)


def test_artifacts_reject_secret_environment_fields(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ArtifactError, match="secret-like"):
        store.begin(
            task=_task(),
            trial=_trial(),
            environment={"api_key": "must-not-persist"},
        )


def test_artifacts_allow_non_secret_auth_presence_flag(tmp_path: Path) -> None:
    paths = ArtifactStore(tmp_path).begin(
        task=_task(),
        trial=_trial(),
        environment={"provider_auth_configured": True},
    )

    assert paths.resolve(paths.manifest.environment).is_file()
