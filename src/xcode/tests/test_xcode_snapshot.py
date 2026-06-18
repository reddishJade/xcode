from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from xcode.harness.snapshot import (
    MAX_SNAPSHOT_FILE_BYTES,
    ChangeEntry,
    SkippedFileInfo,
    SnapshotService,
    SnapshotStore,
    SnapshotUnsupportedError,
    TurnSnapshotRecord,
)


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *cmd], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _hidden_git(project_root: Path, session_id: str, cmd: list[str]) -> str:
    git_dir = project_root / ".local" / "snapshots" / session_id / ".git"
    result = subprocess.run(
        ["git", "--git-dir", str(git_dir), "--work-tree", str(project_root), *cmd],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_git_project(root: Path) -> None:
    _git(["init"], root)
    _git(["config", "user.name", "test"], root)
    _git(["config", "user.email", "test@test"], root)
    (root / ".gitignore").write_text("*.log\n")
    _git(["add", ".gitignore"], root)
    _git(["commit", "-m", "init"], root)


class TestSnapshotService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        _init_git_project(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _user_index_hash(self) -> str:
        index = self.root / ".git" / "index"
        if not index.exists():
            return ""
        return hashlib.sha256(index.read_bytes()).hexdigest()

    def test_track_returns_tree_hash_not_commit(self) -> None:
        svc = SnapshotService(self.root, "test-1")
        result = svc.track()
        self.assertEqual(len(result.snapshot_id), 40)
        # Verify it is a tree object, not a commit
        obj_type = _git(
            ["cat-file", "-t", result.snapshot_id],
            cwd=self.root,
        )
        self.assertEqual(obj_type, "tree")

    def test_hidden_git_init_does_not_create_project_git(self) -> None:
        user_git = self.root / ".git"
        pre_inode = user_git.stat().st_ino
        SnapshotService(self.root, "test-2")
        post_inode = user_git.stat().st_ino
        self.assertEqual(pre_inode, post_inode)
        self.assertTrue(user_git.is_dir())

    def test_track_does_not_mutate_user_git_index(self) -> None:
        svc = SnapshotService(self.root, "test-3")
        pre_hash = self._user_index_hash()
        svc.track()
        post_hash = self._user_index_hash()
        self.assertEqual(pre_hash, post_hash)

    def test_undo_restore_does_not_mutate_user_git_index(self) -> None:
        (self.root / "hello.txt").write_text("world")
        svc = SnapshotService(self.root, "test-4")
        pre = svc.track()
        (self.root / "hello.txt").write_text("modified")
        svc.track()
        pre_idx = self._user_index_hash()
        svc.restore_file(pre.snapshot_id, "hello.txt")
        post_idx = self._user_index_hash()
        self.assertEqual(pre_idx, post_idx)
        self.assertEqual((self.root / "hello.txt").read_text(), "world")

    def test_no_git_stash_or_reset_used(self) -> None:
        source = Path(SnapshotService.__module__)
        code = source.read_text() if source.exists() else ""
        # Search for forbidden command invocations
        for forbidden in ["stash", "reset"]:
            # Only flag if it appears without --git-dir override
            lines = [line for line in code.splitlines() if forbidden in line.lower()]
            for line in lines:
                self.assertIn(
                    "--git-dir",
                    line,
                    msg=f"forbidden git command without --git-dir: {line}",
                )

    def test_pre_post_snapshots_around_turn(self) -> None:
        (self.root / "a.txt").write_text("initial")
        svc = SnapshotService(self.root, "test-5")
        pre = svc.track()
        self.assertIsInstance(pre.snapshot_id, str)
        self.assertEqual(len(pre.snapshot_id), 40)

        (self.root / "a.txt").write_text("changed")
        post = svc.track()
        self.assertNotEqual(pre.snapshot_id, post.snapshot_id)

    def test_diff_tree_modified(self) -> None:
        (self.root / "test.txt").write_text("original")
        svc = SnapshotService(self.root, "test-6")
        pre = svc.track()
        (self.root / "test.txt").write_text("content after")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        self.assertIn(ChangeEntry(path="test.txt", kind="modified"), changes)

    def test_diff_tree_created(self) -> None:
        svc = SnapshotService(self.root, "test-7")
        pre = svc.track()
        (self.root / "new_file.py").write_text("x = 1")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        self.assertIn(ChangeEntry(path="new_file.py", kind="created"), changes)

    def test_diff_tree_deleted(self) -> None:
        (self.root / "deleteme.txt").write_text("bye")
        svc = SnapshotService(self.root, "test-8")
        pre = svc.track()
        (self.root / "deleteme.txt").unlink()
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        self.assertIn(ChangeEntry(path="deleteme.txt", kind="deleted"), changes)

    def test_env_secret_excluded(self) -> None:
        (self.root / ".env").write_text("SECRET=1")
        (self.root / ".env.local").write_text("LOCAL=1")
        svc = SnapshotService(self.root, "test-env")
        result = svc.track()
        skipped_paths = [s.path for s in result.skipped_files]
        self.assertIn(".env", skipped_paths)
        self.assertIn(".env.local", skipped_paths)

    def test_env_example_included(self) -> None:
        (self.root / ".env.example").write_text("EXAMPLE=1")
        svc = SnapshotService(self.root, "test-env-example")
        result = svc.track()
        skipped_paths = [s.path for s in result.skipped_files]
        # Verify it is NOT in skipped files
        self.assertNotIn(".env.example", skipped_paths)
        # Verify it IS in the tree (via hidden git dir)
        tree_contents = _hidden_git(
            self.root,
            "test-env-example",
            ["ls-tree", "--name-only", result.snapshot_id],
        )
        self.assertIn(".env.example", tree_contents)

    def test_normal_dotfiles_tracked(self) -> None:
        (self.root / ".editorconfig").write_text("root = true")
        (self.root / ".prettierrc").write_text("{}")
        (self.root / ".ruff.toml").write_text("line-length = 88")
        svc = SnapshotService(self.root, "test-dotfiles")
        result = svc.track()
        tree_contents = _hidden_git(
            self.root,
            "test-dotfiles",
            ["ls-tree", "--name-only", result.snapshot_id],
        )
        self.assertIn(".editorconfig", tree_contents)
        self.assertIn(".prettierrc", tree_contents)
        self.assertIn(".ruff.toml", tree_contents)

    def test_large_files_skipped(self) -> None:
        large = "x" * (MAX_SNAPSHOT_FILE_BYTES + 1)
        (self.root / "large.bin").write_text(large)
        svc = SnapshotService(self.root, "test-large")
        result = svc.track()
        skipped_paths = [s.path for s in result.skipped_files]
        self.assertIn("large.bin", skipped_paths)
        self.assertTrue(any("too large" in s.reason for s in result.skipped_files))

    def test_structural_exclusions_applied(self) -> None:
        (self.root / "node_modules" / "pkg" / "index.js").parent.mkdir(
            parents=True, exist_ok=True
        )
        (self.root / "node_modules" / "pkg" / "index.js").write_text("x")
        (self.root / "__pycache__" / "cache.pyc").parent.mkdir(
            parents=True, exist_ok=True
        )
        (self.root / "__pycache__" / "cache.pyc").write_text("bytes")
        svc = SnapshotService(self.root, "test-excl")
        result = svc.track()
        tree_contents = _hidden_git(
            self.root,
            "test-excl",
            ["ls-tree", "--name-only", result.snapshot_id],
        )
        self.assertNotIn("node_modules/pkg/index.js", tree_contents)
        self.assertNotIn("__pycache__/cache.pyc", tree_contents)

    def test_dangerous_paths_rejected(self) -> None:
        svc = SnapshotService(self.root, "test-safe")
        with self.assertRaisesRegex(ValueError, "absolute path"):
            svc._validate_path("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "parent-directory"):
            svc._validate_path("../outside.txt")
        with self.assertRaisesRegex(ValueError, "empty path"):
            svc._validate_path("")

    def test_undo_restores_modified_file(self) -> None:
        (self.root / "restore.txt").write_text("original")
        svc = SnapshotService(self.root, "test-restore")
        pre = svc.track()
        (self.root / "restore.txt").write_text("modified")
        svc.restore_file(pre.snapshot_id, "restore.txt")
        self.assertEqual((self.root / "restore.txt").read_text(), "original")

    def test_conflict_detection(self) -> None:
        (self.root / "c.txt").write_text("start")
        svc = SnapshotService(self.root, "test-conflict")
        svc.track()
        (self.root / "c.txt").write_text("after turn")
        post = svc.track()
        # No conflict: current matches post
        self.assertFalse(svc.has_conflict(post.snapshot_id, "c.txt"))
        # Introduce conflict: modify after post snapshot
        (self.root / "c.txt").write_text("manual edit")
        self.assertTrue(svc.has_conflict(post.snapshot_id, "c.txt"))

    def test_unrelated_files_untouched(self) -> None:
        (self.root / "a.txt").write_text("a")
        (self.root / "b.txt").write_text("b")
        svc = SnapshotService(self.root, "test-unrelated")
        pre = svc.track()
        (self.root / "a.txt").write_text("a modified")
        post = svc.track()
        # Only a.txt changed
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].path, "a.txt")
        # b.txt is untouched
        self.assertEqual((self.root / "b.txt").read_text(), "b")

    def test_non_git_project_unsupported(self) -> None:
        non_git = self.root.parent / "non_git_project"
        non_git.mkdir(exist_ok=True)
        with self.assertRaises(SnapshotUnsupportedError):
            SnapshotStore(non_git)


class TestSnapshotStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        _init_git_project(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_turn_records_in_index_json(self) -> None:
        store = SnapshotStore(self.root)
        svc = store.service("sess-1")
        pre = svc.track()
        (self.root / "data.txt").write_text("content")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        store.record_turn(
            session_id="sess-1",
            turn_id="001",
            pre_snapshot_id=pre.snapshot_id,
            post_snapshot_id=post.snapshot_id,
            changed_files=changes,
        )
        records = store.list_records("sess-1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].turn_id, "001")
        self.assertEqual(records[0].changed_files[0].path, "data.txt")
        self.assertEqual(records[0].changed_files[0].kind, "created")

    def test_get_undoable_records_lifo(self) -> None:
        store = SnapshotStore(self.root)
        svc = store.service("sess-lifo")
        for i in range(3):
            pre = svc.track()
            (self.root / f"turn_{i}.txt").write_text(f"content_{i}")
            post = svc.track()
            changes = svc.diff(pre.snapshot_id, post.snapshot_id)
            turn_id = f"{i + 1:03d}"
            store.record_turn(
                session_id="sess-lifo",
                turn_id=turn_id,
                pre_snapshot_id=pre.snapshot_id,
                post_snapshot_id=post.snapshot_id,
                changed_files=changes,
            )
        # After 3 turns, undo 2 should return latest 2 in order
        undoable = store.get_undoable_records("sess-lifo", 2)
        self.assertEqual(len(undoable), 2)
        self.assertEqual(undoable[0].turn_id, "002")
        self.assertEqual(undoable[1].turn_id, "003")

    def test_undo_mark_and_skip(self) -> None:
        store = SnapshotStore(self.root)
        svc = store.service("sess-skip")
        pre = svc.track()
        (self.root / "s.txt").write_text("data")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        store.record_turn(
            session_id="sess-skip",
            turn_id="001",
            pre_snapshot_id=pre.snapshot_id,
            post_snapshot_id=post.snapshot_id,
            changed_files=changes,
        )
        records = store.get_undoable_records("sess-skip", 1)
        self.assertEqual(len(records), 1)
        records[0].undone = True
        store.update_record("sess-skip", records[0])
        self.assertEqual(len(store.get_undoable_records("sess-skip", 1)), 0)

    def test_skipped_files_stored_in_record(self) -> None:
        store = SnapshotStore(self.root)
        svc = store.service("sess-skp")
        pre = svc.track()
        (self.root / "u.txt").write_text("ok")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        store.record_turn(
            session_id="sess-skp",
            turn_id="001",
            pre_snapshot_id=pre.snapshot_id,
            post_snapshot_id=post.snapshot_id,
            changed_files=changes,
            skipped_files=[SkippedFileInfo("big.bin", "too large")],
        )
        records = store.list_records("sess-skp")
        self.assertEqual(len(records[0].skipped_files), 1)
        self.assertEqual(records[0].skipped_files[0].path, "big.bin")

    def test_tool_names_stored_and_round_trip(self) -> None:
        """验证 TurnSnapshotRecord 的 tool_names 字段持久化往返。"""
        store = SnapshotStore(self.root)
        svc = store.service("sess-tools")
        pre = svc.track()
        (self.root / "a.txt").write_text("a")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        store.record_turn(
            session_id="sess-tools",
            turn_id="001",
            pre_snapshot_id=pre.snapshot_id,
            post_snapshot_id=post.snapshot_id,
            changed_files=changes,
            tool_names=["read_file", "write_file"],
        )
        records = store.list_records("sess-tools")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].tool_names, ["read_file", "write_file"])

        # 验证 JSON 往返
        data = records[0].to_dict()
        restored = TurnSnapshotRecord.from_dict(data)
        self.assertEqual(restored.tool_names, ["read_file", "write_file"])

    def test_tool_names_default_empty(self) -> None:
        """验证不传 tool_names 时默认为空列表。"""
        store = SnapshotStore(self.root)
        svc = store.service("sess-tools-def")
        pre = svc.track()
        (self.root / "b.txt").write_text("b")
        post = svc.track()
        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        store.record_turn(
            session_id="sess-tools-def",
            turn_id="001",
            pre_snapshot_id=pre.snapshot_id,
            post_snapshot_id=post.snapshot_id,
            changed_files=changes,
        )
        records = store.list_records("sess-tools-def")
        self.assertEqual(records[0].tool_names, [])


class TestSnapshotDeletedFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        _init_git_project(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_delete_created_file(self) -> None:
        (self.root / "new.txt").write_text("created during turn")
        svc = SnapshotService(self.root, "test-del")
        pre = svc.track()
        (self.root / "new.txt").unlink()
        post = svc.track()

        changes = svc.diff(pre.snapshot_id, post.snapshot_id)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].kind, "deleted")
        self.assertEqual(changes[0].path, "new.txt")


class TestActionExtractorDeleteFile(unittest.TestCase):
    def test_delete_file_extracts_write_action(self) -> None:
        from xcode.harness.observability.permission_model import (
            ActionExtractor,
        )

        action = ActionExtractor().extract("delete_file", {"path": "test.txt"})
        self.assertEqual(action.tool, "delete_file")
        self.assertEqual(action.capability, "write")
        self.assertEqual(action.operation, "delete_file")
        self.assertEqual(len(action.targets), 1)
        self.assertEqual(action.targets[0].kind, "path")
        self.assertEqual(action.targets[0].access, "write")
        self.assertEqual(action.targets[0].value, "test.txt")


if __name__ == "__main__":
    unittest.main()
