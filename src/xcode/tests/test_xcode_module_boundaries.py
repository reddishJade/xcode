"""模块边界回归测试。"""

from __future__ import annotations

from pathlib import Path
import pytest


class XcodeModuleBoundaryTests:
    """验证实验能力与稳定模块边界。"""

    def test_experimental_package_owns_optional_features(self) -> None:
        """仍在实验阶段的编排能力必须位于 experimental package。"""
        package_root = Path(__file__).resolve().parents[1]
        experimental_root = package_root / "experimental"
        assert experimental_root.is_dir()
        assert {
            "mailbox.py",
            "orchestration_store.py",
            "task_progress.py",
            "task_store.py",
        } <= {path.name for path in experimental_root.glob("*.py")}

        assert not (package_root / "harness" / "mailbox.py").exists()
        assert not (package_root / "harness" / "task_store.py").exists()
        assert not (package_root / "harness" / "task_progress.py").exists()
        assert not (package_root / "harness" / "orchestration_store.py").exists()
        assert not (package_root / "coding_agent" / "tools" / "worktree.py").exists()

    def test_worktree_backend_is_harness_infra(self) -> None:
        """worktree 隔离是 subagent 的正式 harness 依赖。"""
        package_root = Path(__file__).resolve().parents[1]
        assert (package_root / "harness" / "worktree.py").exists()
        assert not (package_root / "experimental" / "worktree.py").exists()


if __name__ == "__main__":
    pytest.main()
