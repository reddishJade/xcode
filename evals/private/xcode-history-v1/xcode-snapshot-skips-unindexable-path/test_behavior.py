"""Agent 不可见的 snapshot 局部 Git add 失败行为 oracle。"""

from pathlib import Path
import subprocess
import threading
from unittest.mock import patch

from xcode.harness.snapshot import SNAPSHOT_EXCLUDES, SnapshotService


def _completed(args: list[str], returncode: int = 0, stdout: str = ""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")


def test_one_unindexable_path_does_not_abort_the_snapshot(tmp_path: Path) -> None:
    service = object.__new__(SnapshotService)
    service._project_root = tmp_path
    service._lock = threading.Lock()
    service._skipped = []
    added: list[str] = []

    def fake_git(
        args: list[str],
        check: bool = True,
        timeout: int = 30,
        input_text: str | None = None,
    ):
        del check, timeout, input_text
        if args == ["write-tree"]:
            return _completed(args, stdout="tree-id\n")
        if len(args) == 3 and args[:2] == ["--literal-pathspecs", "add"]:
            added.append(args[2])
            return _completed(args, returncode=1 if args[2] == "nul" else 0)
        return _completed(args)

    with (
        patch.object(service, "_enumerate_files", return_value=["ok.py", "nul"]),
        patch.object(service, "_git", side_effect=fake_git),
    ):
        result = service.track()

    assert result.snapshot_id == "tree-id"
    assert added == ["ok.py", "nul"]
    assert [(item.path, item.reason) for item in result.skipped_files] == [
        ("nul", "skipped: git add failed (reserved name?)")
    ]


def test_reserved_nul_forms_are_proactively_excluded() -> None:
    assert "nul" in SNAPSHOT_EXCLUDES
    assert "NUL" in SNAPSHOT_EXCLUDES
    assert "NUL.*" in SNAPSHOT_EXCLUDES
