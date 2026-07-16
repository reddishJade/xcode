"""bubblewrap mount 边界契约测试。"""

from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from xcode.evals.isolation import BubblewrapExecutor, IsolationError
from xcode.evals.schema import (
    ModelConfig,
    ResourceBudget,
    Task,
    TaskSource,
    Trial,
    Variant,
)
from xcode.harness.config import XcodeRuntimeConfig


def _task_and_trial() -> tuple[Task, Trial]:
    task = Task(
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
        allowed_paths=(".",),
        budget=ResourceBudget(wall_time_seconds=30, model_calls=5, tool_calls=20),
    )
    trial = Trial(
        trial_id="trial-1",
        experiment_id="experiment-1",
        task_id=task.task_id,
        dataset_version=task.dataset_version,
        variant=Variant(
            variant_id="full",
            harness_revision="current",
            capabilities={"full": True},
        ),
        model=ModelConfig(provider="real", model="real-model"),
        budget=task.budget,
        repetition=0,
        workspace_revision=task.source.revision,
        command=("xcode-eval", "run"),
    )
    return task, trial


def test_bubblewrap_mounts_runtime_but_not_control_repository(tmp_path: Path) -> None:
    source = tmp_path / "source/xcode"
    virtualenv = tmp_path / "venv"
    runtime = tmp_path / "python"
    uv_executable = tmp_path / "uv"
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    for path in (source, virtualenv, runtime, workspace, output):
        path.mkdir(parents=True)
    uv_executable.touch()
    executor = BubblewrapExecutor(
        xcode_source=source,
        virtualenv=virtualenv,
        python_runtime=runtime,
        uv_executable=uv_executable,
    )

    command = executor._command(workspace=workspace, output=output)

    assert "--clearenv" in command
    assert str(source.resolve()) in command
    assert str(workspace.resolve()) in command
    assert str(output.resolve()) in command
    assert "/runtime/xcode" in command
    assert "/run/systemd/resolve" in command
    assert command[command.index("VIRTUAL_ENV") + 1] == str(virtualenv.resolve())
    assert str(uv_executable.resolve()) in command
    assert f"/tools:{virtualenv.resolve()}/bin:/usr/bin:/bin" in command
    assert "evals/private" not in " ".join(command)
    assert "xcode.config.json" not in " ".join(command)


def test_bubblewrap_accepts_precreated_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    output.mkdir(parents=True, exist_ok=True)

    assert output.is_dir()


def test_bubblewrap_converts_timeout_to_isolation_error(tmp_path: Path) -> None:
    source = tmp_path / "source/xcode"
    virtualenv = tmp_path / "venv"
    runtime = tmp_path / "python"
    uv_executable = tmp_path / "uv"
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    for path in (source, virtualenv, runtime, workspace, output):
        path.mkdir(parents=True)
    uv_executable.touch()
    executor = BubblewrapExecutor(
        xcode_source=source,
        virtualenv=virtualenv,
        python_runtime=runtime,
        uv_executable=uv_executable,
    )
    task, trial = _task_and_trial()

    with patch(
        "xcode.evals.isolation.subprocess.run",
        side_effect=subprocess.TimeoutExpired("bwrap", 60),
    ):
        with pytest.raises(IsolationError, match="exceeded wall-clock timeout"):
            executor.run(
                task=task,
                trial=trial,
                runtime_config=XcodeRuntimeConfig(),
                workspace=workspace,
                output=output,
            )
