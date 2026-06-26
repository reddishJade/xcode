from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xcode.experimental.orchestration_store import OrchestrationStore
from xcode.experimental.task_progress import (
    build_progress_tools,
    expire_stale_runs,
    resume_run,
    resume_task,
    retry_run,
    save_progress,
    start_run,
)
from xcode.experimental.task_store import TaskStore
import pytest


class TestTaskProgress:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = TaskStore(self.root)
        self.orchestration = OrchestrationStore(self.root)

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_save_and_resume_progress_basic(self) -> None:
        task = self.store.create("Implement login feature")
        assert task.id == 1

        checklist = [
            {"step": 1, "title": "Design schema", "status": "completed"},
            {"step": 2, "title": "Implement API", "status": "in_progress"},
            {"step": 3, "title": "Write unit tests", "status": "pending"},
        ]

        save_progress(self.store, task.id, checklist)

        updated_task = self.store.get(task.id)
        saved_checklist = updated_task.payload.get("feature_list")
        assert saved_checklist == checklist

        progress_file = self.root / ".local" / "progress_summary.md"
        assert progress_file.exists()

        content = progress_file.read_text(encoding="utf-8")
        assert "Progress: 33.3%" in content
        assert "Step 1: Design schema" in content
        assert "- [/] Step 2: Implement API" in content
        assert "- [ ] Step 3: Write unit tests" in content
        assert "Current Active Step:\n- Implement API" in content

        resumed_checklist = resume_task(self.store, task.id)
        assert resumed_checklist == checklist

    def test_save_progress_merges_not_overwrites_payload(self) -> None:
        """save_progress 必须保留 payload 中的其他字段（如 blocked_by）。"""
        task = self.store.create("task with deps", {"blocked_by": [5]})
        save_progress(self.store, task.id, [{"title": "step1", "status": "pending"}])
        payload = self.store.get(task.id).payload
        assert payload.get("blocked_by") == [5]
        assert payload.get("feature_list") == [{"title": "step1", "status": "pending"}]

    def test_save_progress_custom_summary_path(self) -> None:
        """summary_path 可配置。"""
        task = self.store.create("task")
        custom = self.root / "custom" / "summary.md"
        save_progress(
            self.store,
            task.id,
            [{"title": "x", "status": "completed"}],
            summary_path=custom,
        )
        assert custom.exists()
        assert (self.root / ".local" / "progress_summary.md").exists() is False

    def test_build_progress_tools_uses_configured_summary_path(self) -> None:
        """build_progress_tools 的 summary_path 参数透传到 save_progress handler。"""
        task = self.store.create("task")
        custom = self.root / "configured" / "progress.md"
        tools = build_progress_tools(
            self.store, self.orchestration, summary_path=custom
        )
        save_tool = next(t for t in tools if t.name == "save_task_progress")
        save_tool.handler(
            {
                "task_id": task.id,
                "feature_list": [{"title": "x", "status": "completed"}],
            }
        )
        assert custom.exists()
        assert (self.root / ".local" / "progress_summary.md").exists() is False

    def test_resume_missing_or_unknown_task(self) -> None:
        resumed = resume_task(self.store, 999)
        assert resumed == []

        task = self.store.create("Some other task")
        resumed_empty = resume_task(self.store, task.id)
        assert resumed_empty == []

    def test_progress_tools_basic_flow(self) -> None:
        task = self.store.create("Implement search")
        tools = {
            tool.name: tool
            for tool in build_progress_tools(self.store, self.orchestration)
        }

        saved = tools["save_task_progress"].handler(
            {
                "task_id": 1,
                "feature_list": [
                    {"title": "Index files", "status": "completed"},
                ],
            }
        )
        assert saved == f"saved progress for task {task.id}"

        resumed = tools["resume_task_progress"].handler({"task_id": 1})
        assert '"title": "Index files"' in resumed
        assert '"status": "completed"' in resumed
        assert (self.root / ".local" / "progress_summary.md").exists()

    def test_start_run_dispatches_subtasks_and_can_resume(self) -> None:
        task = self.store.create("Implement orchestration")

        state = start_run(
            self.store,
            self.orchestration,
            task.id,
            timeout_seconds=60,
            retry_limit=1,
            subtasks=["Write tests", "Update docs"],
        )
        resumed = resume_run(self.store, self.orchestration, task.id)
        child_titles = [self.store.get(tid).title for tid in state.subtask_ids]

        assert state.status == "running"
        assert state.attempt == 1
        assert resumed == state
        assert child_titles == ["Write tests", "Update docs"]
        assert self.store.get(task.id).status == "claimed"

    def test_start_run_writes_orchestration_file_not_payload(self) -> None:
        """orchestration 状态写入独立文件，不混入 task.payload。"""
        task = self.store.create("task")
        start_run(self.store, self.orchestration, task.id, timeout_seconds=60)
        # payload 不应含 orchestration 键
        payload = self.store.get(task.id).payload
        assert "orchestration" not in payload
        # 独立文件存在
        orch_path = self.root / ".local" / "orchestration" / f"{task.id}.json"
        assert orch_path.exists()

    def test_resume_run_reads_from_independent_file(self) -> None:
        """resume_run 从独立文件读取，与 start_run 写入的 state 一致。"""
        task = self.store.create("task")
        state = start_run(
            self.store, self.orchestration, task.id, timeout_seconds=60, retry_limit=2
        )
        # 新建 orchestration store 模拟重启
        orch2 = OrchestrationStore(self.root)
        resumed = resume_run(self.store, orch2, task.id)
        assert resumed == state

    def test_save_progress_preserves_orchestration(self) -> None:
        """save_progress 不破坏 orchestration 状态（独立存储）。"""
        task = self.store.create("task")
        state = start_run(
            self.store, self.orchestration, task.id, timeout_seconds=60, retry_limit=1
        )
        save_progress(self.store, task.id, [{"title": "step1", "status": "completed"}])
        resumed_state = resume_run(self.store, self.orchestration, task.id)
        assert resumed_state == state
        # task status 仍为 claimed（save_progress 不改 status）
        assert self.store.get(task.id).status == "claimed"

    def test_expire_stale_runs_releases_claimed_task(self) -> None:
        task = self.store.create("Long task")
        start_run(
            self.store,
            self.orchestration,
            task.id,
            timeout_seconds=1,
            retry_limit=1,
        )
        # 手动写过期 lease 到独立文件
        from dataclasses import asdict

        state = self.orchestration.get(task.id)
        assert state is not None
        expired_state = type(state)(
            **{**asdict(state), "lease_expires_at": "2000-01-01T00:00:00Z"}
        )
        self.orchestration.set(expired_state)

        expired = expire_stale_runs(self.store, self.orchestration)

        assert len(expired) == 1
        assert expired[0].status == "timed_out"
        assert self.store.get(task.id).status == "pending"

    def test_expire_stale_uses_lease_index_avoids_full_scan(self) -> None:
        """expire_stale_runs 通过 lease 索引查找，不遍历 TaskStore.list()。"""
        import unittest.mock as mock

        task = self.store.create("Long task")
        start_run(
            self.store, self.orchestration, task.id, timeout_seconds=1, retry_limit=1
        )
        from dataclasses import asdict

        state = self.orchestration.get(task.id)
        assert state is not None
        expired_state = type(state)(
            **{**asdict(state), "lease_expires_at": "2000-01-01T00:00:00Z"}
        )
        self.orchestration.set(expired_state)

        # spy TaskStore.list：若被调用说明回退到全表扫描
        with mock.patch.object(self.store, "list", wraps=self.store.list) as spy_list:
            expired = expire_stale_runs(self.store, self.orchestration)
            assert spy_list.call_count == 0
        assert len(expired) == 1

    def test_retry_run_respects_retry_limit(self) -> None:
        task = self.store.create("Retry task")
        start_run(
            self.store,
            self.orchestration,
            task.id,
            timeout_seconds=60,
            retry_limit=1,
        )

        retried = retry_run(self.store, self.orchestration, task.id)

        assert retried.attempt == 2
        with pytest.raises(ValueError):
            retry_run(self.store, self.orchestration, task.id)

    def test_resume_run_warns_on_missing_state(self, caplog) -> None:
        """无 orchestration 文件时记录 warning 并返回 default state。"""
        import logging

        task = self.store.create("task")
        with caplog.at_level(logging.WARNING, logger="xcode.experimental.task_progress"):
            state = resume_run(self.store, self.orchestration, task.id)
        assert state.status == "pending"  # 回退到 task.status
        assert state.attempt == 0
        assert any("no orchestration state" in r.message for r in caplog.records)

    def test_resume_run_warns_on_missing_fields(self, caplog) -> None:
        """orchestration 文件缺字段时记录 warning。"""
        import logging

        task = self.store.create("task")
        # 手动写一个缺 lease_expires_at 的 state 文件
        orch_path = self.root / ".local" / "orchestration" / f"{task.id}.json"
        orch_path.parent.mkdir(parents=True, exist_ok=True)
        orch_path.write_text(
            json.dumps(
                {
                    "task_id": task.id,
                    "status": "running",
                    "attempt": 1,
                    "retry_limit": 0,
                    "lease_expires_at": "",
                    "subtask_ids": [],
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="xcode.experimental.task_progress"):
            state = resume_run(self.store, self.orchestration, task.id)
        assert state.status == "running"
        assert any("missing fields" in r.message for r in caplog.records)

    def test_lease_index_rebuilt_on_corruption(self, caplog) -> None:
        """lease 索引损坏时回退全表扫描并重建。"""
        import logging

        task = self.store.create("task")
        start_run(
            self.store, self.orchestration, task.id, timeout_seconds=1, retry_limit=1
        )
        from dataclasses import asdict

        state = self.orchestration.get(task.id)
        assert state is not None
        expired_state = type(state)(
            **{**asdict(state), "lease_expires_at": "2000-01-01T00:00:00Z"}
        )
        self.orchestration.set(expired_state)

        # 写坏索引
        index_path = self.root / ".local" / "orchestration" / ".lease_index"
        index_path.write_text("{invalid json", encoding="utf-8")

        with caplog.at_level(
            logging.WARNING, logger="xcode.experimental.orchestration_store"
        ):
            expired_ids = self.orchestration.list_expired()
        assert task.id in expired_ids
        assert any(
            "full scan" in r.message or "corrupted" in r.message for r in caplog.records
        )
        # 索引被重建
        rebuilt = json.loads(index_path.read_text(encoding="utf-8"))
        assert str(task.id) in rebuilt

    def test_progress_tools_expose_orchestration_flow(self) -> None:
        task = self.store.create("Tool orchestration")
        tools = {
            tool.name: tool
            for tool in build_progress_tools(self.store, self.orchestration)
        }

        started = tools["start_task_run"].handler(
            {
                "task_id": task.id,
                "timeout_seconds": 60,
                "retry_limit": 1,
                "subtasks": ["Child"],
            }
        )
        resumed = tools["resume_task_run"].handler({"task_id": task.id})

        assert '"status": "running"' in started
        assert '"subtask_ids"' in resumed
        assert "expire_task_runs" in tools


if __name__ == "__main__":
    pytest.main()
