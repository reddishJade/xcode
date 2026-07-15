"""SWE-bench Lite 小样本 adapter 的公开数据与 prediction 边界测试。"""

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from xcode.evals.artifacts import ArtifactStore
from xcode.evals.cli import _verifier_spec
from xcode.evals.schema import (
    ErrorCategory,
    ResourceBudget,
    ResourceUsage,
    Trial,
    TrialResult,
    Variant,
)
from xcode.evals.swebench_lite import (
    SWEbenchLiteError,
    build_fast_command,
    export_predictions,
    load_instances_jsonl,
    to_task,
)


def _budget() -> ResourceBudget:
    return ResourceBudget(wall_time_seconds=60, model_calls=10, tool_calls=20)


def test_instance_conversion_keeps_only_public_fields(tmp_path: Path) -> None:
    source = tmp_path / "instances.jsonl"
    source.write_text(
        json.dumps(
            {
                "instance_id": "pallets__flask-123",
                "repo": "pallets/flask",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the documented behavior.",
                "test_patch": "hidden material",
                "FAIL_TO_PASS": "hidden material",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    instance = load_instances_jsonl(source)[0]
    task = to_task(
        instance,
        dataset_version="swebench-lite-smoke-v1",
        license_name="MIT",
        budget=_budget(),
    )

    payload = task.model_dump_json()
    assert task.source.repository == "https://github.com/pallets/flask.git"
    assert task.source.revision == "a" * 40
    assert "hidden material" not in payload
    assert task.verifier_id == "swebench-lite-fast"


def test_export_prediction_requires_complete_sealed_trial(tmp_path: Path) -> None:
    instance_path = tmp_path / "instances.jsonl"
    instance_path.write_text(
        json.dumps(
            {
                "instance_id": "pallets__flask-123",
                "repo": "pallets/flask",
                "base_commit": "b" * 40,
                "problem_statement": "Fix it.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task = to_task(
        load_instances_jsonl(instance_path)[0],
        dataset_version="swebench-lite-smoke-v1",
        license_name="MIT",
        budget=_budget(),
    )
    trial = Trial(
        trial_id="swe-smoke.pallets__flask-123.full.r0",
        experiment_id="swe-smoke",
        task_id=task.task_id,
        dataset_version=task.dataset_version,
        variant=Variant(variant_id="full", harness_revision="c" * 40, capabilities={}),
        model={"provider": "openai", "model": "example-model"},
        budget=task.budget,
        repetition=0,
        workspace_revision=task.source.revision,
        command=("xcode-eval",),
    )
    store = ArtifactStore(tmp_path / "artifacts")
    paths = store.begin(task=task, trial=trial, environment={"platform": "test"})
    paths.resolve(paths.manifest.trace).touch()
    paths.resolve(paths.manifest.verifier_log).touch()
    paths.resolve(paths.manifest.patch).write_text(
        "diff --git a/app.py b/app.py\n", encoding="utf-8"
    )
    now = datetime.now(UTC)
    result = TrialResult(
        trial_id=trial.trial_id,
        started_at=now,
        finished_at=now,
        agent_completed=True,
        valid_trial=False,
        verifier=None,
        error_category=ErrorCategory.VERIFIER_FAILURE,
        error_message="external scorer pending",
        termination_reason="verifier_failure",
        usage=ResourceUsage(wall_time_seconds=1, model_calls=1, tool_calls=1),
        artifacts=paths.manifest,
    )
    store.finish(paths=paths, patch="diff --git a/app.py b/app.py\n", result=result)

    output = tmp_path / "predictions.jsonl"
    predictions = export_predictions((paths.root,), output=output)

    assert predictions[0].instance_id == "pallets__flask-123"
    assert predictions[0].model_name_or_path == "example-model"
    assert json.loads(output.read_text(encoding="utf-8"))["model_patch"].startswith(
        "diff --git"
    )


def test_export_rejects_duplicate_instance_predictions(tmp_path: Path) -> None:
    with pytest.raises(SWEbenchLiteError, match="no Trial artifacts"):
        export_predictions((), output=tmp_path / "predictions.jsonl")


def test_fast_command_matches_reference_shape(tmp_path: Path) -> None:
    command = build_fast_command(
        binary=tmp_path / "swe-bench-fast",
        dataset=tmp_path / "dataset.jsonl",
        predictions=tmp_path / "predictions.jsonl",
        run_id="xcode-lite-smoke",
        output=tmp_path / "report.json",
    )

    assert command == (
        str(tmp_path / "swe-bench-fast"),
        "run",
        "--dataset",
        str(tmp_path / "dataset.jsonl"),
        "--predictions",
        str(tmp_path / "predictions.jsonl"),
        "--workers",
        "1",
        "--timeout",
        "900",
        "--run-id",
        "xcode-lite-smoke",
        "--output",
        str(tmp_path / "report.json"),
    )


def test_lite_task_uses_patch_aware_external_verifier(tmp_path: Path) -> None:
    source = tmp_path / "instances.jsonl"
    source.write_text(
        json.dumps(
            {
                "instance_id": "pallets__flask-123",
                "repo": "pallets/flask",
                "base_commit": "c" * 40,
                "problem_statement": "Fix it.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task = to_task(
        load_instances_jsonl(source)[0],
        dataset_version="swebench-lite-smoke-v1",
        license_name="MIT",
        budget=_budget(),
    )

    spec = _verifier_spec(
        control_root=tmp_path / "control",
        private_root=tmp_path / "private",
        task=task,
    )

    assert spec.version == "swebench-3.0.11"
    assert spec.command[-1] == "{patch}"
    assert spec.timeout_seconds == 300
