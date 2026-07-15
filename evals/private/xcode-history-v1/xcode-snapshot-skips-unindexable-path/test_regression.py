"""Snapshot 父版本已有的正常文件与 secret 排除回归。"""

from pathlib import Path

from xcode.harness.snapshot import SnapshotService


def test_normal_snapshot_still_tracks_files_and_skips_env_secret(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value\n", encoding="utf-8")
    service = SnapshotService(tmp_path, "hidden-regression")

    result = service.track()

    assert len(result.snapshot_id) == 40
    assert [(item.path, item.reason) for item in result.skipped_files] == [
        (".env", "excluded: environment secret file")
    ]
