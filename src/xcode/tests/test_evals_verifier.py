"""独立 verifier 边界与结果协议测试。"""

from pathlib import Path

import pytest

from xcode.evals.schema import ResourceBudget, Task, TaskSource, VerifierSpec
from xcode.evals.verifier import VerifierError, VerifierRunner


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
        allowed_paths=("answer.txt",),
        budget=ResourceBudget(wall_time_seconds=60, model_calls=5, tool_calls=20),
    )


def test_verifier_runs_outside_workspace_and_reads_explicit_result(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    hidden = tmp_path / "control" / "hidden"
    workspace.mkdir()
    hidden.mkdir(parents=True)
    script = hidden / "verify.py"
    script.write_text(
        """import json
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
passed = (workspace / "answer.txt").read_text() == "fixed\\n"
Path("verifier-result.json").write_text(json.dumps({
    "resolved": passed,
    "regression_free": True,
    "policy_clean": True,
}))
""",
        encoding="utf-8",
    )
    (workspace / "answer.txt").write_text("fixed\n", encoding="utf-8")
    spec = VerifierSpec(
        verifier_id="hidden-1",
        version="v1",
        command=("python", "verify.py", "{workspace}"),
        hidden_root=str(hidden),
        timeout_seconds=10,
    )

    result = VerifierRunner().run(
        spec=spec,
        task=_task(),
        workspace=workspace,
        changed_paths=("answer.txt",),
        log_path=tmp_path / "artifacts/verifier.log",
    )

    assert result.resolved is True
    assert result.regression_free is True
    assert result.policy_clean is True
    assert result.details["exit_code"] == 0


def test_verifier_rejects_hidden_material_inside_agent_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    hidden = workspace / "hidden"
    hidden.mkdir(parents=True)
    spec = VerifierSpec(
        verifier_id="hidden-1",
        version="v1",
        command=("python", "verify.py"),
        hidden_root=str(hidden),
        timeout_seconds=10,
    )

    with pytest.raises(VerifierError, match="inside the Agent workspace"):
        VerifierRunner().run(
            spec=spec,
            task=_task(),
            workspace=workspace,
            changed_paths=(),
            log_path=tmp_path / "verifier.log",
        )


def test_verifier_missing_result_is_infrastructure_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    hidden = tmp_path / "hidden"
    workspace.mkdir()
    hidden.mkdir()
    spec = VerifierSpec(
        verifier_id="hidden-1",
        version="v1",
        command=("python", "-c", "print('no result')"),
        hidden_root=str(hidden),
        timeout_seconds=10,
    )

    with pytest.raises(VerifierError, match="did not produce"):
        VerifierRunner().run(
            spec=spec,
            task=_task(),
            workspace=workspace,
            changed_paths=(),
            log_path=tmp_path / "verifier.log",
        )


def test_policy_result_is_computed_by_control_plane(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    hidden = tmp_path / "hidden"
    workspace.mkdir()
    hidden.mkdir()
    (hidden / "verify.py").write_text(
        """import json
from pathlib import Path
Path("verifier-result.json").write_text(json.dumps({
    "resolved": True,
    "regression_free": True,
    "policy_clean": True,
}))
""",
        encoding="utf-8",
    )
    spec = VerifierSpec(
        verifier_id="hidden-1",
        version="v1",
        command=("python", "verify.py"),
        hidden_root=str(hidden),
        timeout_seconds=10,
    )

    result = VerifierRunner().run(
        spec=spec,
        task=_task(),
        workspace=workspace,
        changed_paths=("forbidden.txt",),
        log_path=tmp_path / "verifier.log",
    )

    assert result.policy_clean is False
    assert result.details["policy_violations"] == ["forbidden.txt"]
