"""Git workspace 恢复契约测试。"""

from pathlib import Path
import subprocess

import pytest

from xcode.evals.schema import ResourceBudget, Task, TaskSource
from xcode.evals.workspace import (
    changed_paths,
    GitWorkspaceFactory,
    WorkspaceError,
    workspace_digest,
    workspace_patch,
)


def _task(repository: Path, revision: str) -> Task:
    return Task(
        task_id="history-fix",
        dataset_version="v1",
        prompt="修复回归。",
        source=TaskSource(
            kind="git_history",
            repository=str(repository),
            revision=revision,
            license="MIT",
        ),
        verifier_id="history-fix-hidden",
        allowed_paths=("src",),
        budget=ResourceBudget(wall_time_seconds=60, model_calls=5, tool_calls=20),
    )


def test_git_workspace_restores_clean_revision_without_git_metadata(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "eval@example.test"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "Eval"), cwd=repository, check=True)
    source = repository / "src"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "src/value.py"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "initial"), cwd=repository, check=True)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    factory = GitWorkspaceFactory(
        repository=repository,
        workspace_root=tmp_path / "workspaces",
    )

    workspace = factory.create(_task(repository, revision), "trial-1")

    assert (workspace.root / "src/value.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"
    assert not (workspace.root / ".git").exists()
    assert workspace.revision == revision
    assert workspace.initial_digest == workspace_digest(workspace.root)


def test_git_workspace_refuses_reusing_trial_state(tmp_path: Path) -> None:
    repository = Path(__file__).parents[3]
    factory = GitWorkspaceFactory(repository=repository, workspace_root=tmp_path)
    task = _task(repository, "HEAD")
    factory.create(task, "trial-1")

    with pytest.raises(WorkspaceError, match="already exists"):
        factory.create(task, "trial-1")


def test_workspace_records_changed_paths_and_replayable_patch(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    original = root / "value.py"
    original.write_text("VALUE = 1\n", encoding="utf-8")
    initial = {"value.py": original.read_bytes()}
    original.write_text("VALUE = 2\n", encoding="utf-8")
    (root / "new.py").write_text("NEW = True\n", encoding="utf-8")

    assert changed_paths(initial, root) == ("new.py", "value.py")
    patch = workspace_patch(initial, root)
    assert "--- a/value.py" in patch
    assert "+VALUE = 2" in patch
    assert "+++ b/new.py" in patch
