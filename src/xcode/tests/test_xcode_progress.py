from __future__ import annotations

import tempfile
from pathlib import Path

from xcode.harness.task_store import TaskStore
from xcode.harness.task_progress import (
build_progress_tools,
    expire_stale_runs,
    resume_run,
    resume_task,
    retry_run,
    save_progress,
    start_run,
)
import pytest
class TestTaskProgress:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = TaskStore(self.root)

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_save_and_resume_progress_basic(self) -> None:
        # 1. Create a task in the TaskStore
        task = self.store.create("Implement login feature")
        assert task.id == 1

        # 2. Define the sub-tasks (feature list)
        checklist = [
            {"step": 1, "title": "Design schema", "status": "completed"},
            {"step": 2, "title": "Implement API", "status": "in_progress"},
            {"step": 3, "title": "Write unit tests", "status": "pending"},
        ]

        # 3. Save progress
        save_progress(self.store, task.id, checklist)

        # 4. Assert SoT (payload in TaskStore) has been updated under lock protection
        updated_task = self.store.get(task.id)
        saved_checklist = updated_task.payload.get("feature_list")
        assert saved_checklist == checklist

        # 5. Assert derived read-only view 'claude-progress.txt' was written to root
        progress_txt = self.root / "claude-progress.txt"
        assert progress_txt.exists()

        content = progress_txt.read_text(encoding="utf-8")
        assert "Progress: 33.3%" in content
        assert "Step 1: Design schema" in content
        assert "- [/] Step 2: Implement API" in content
        assert "- [ ] Step 3: Write unit tests" in content
        assert "Current Active Step:\n- Implement API" in content

        # 6. Assert resume_task fetches SoT checklist perfectly
        resumed_checklist = resume_task(self.store, task.id)
        assert resumed_checklist == checklist

    def test_resume_missing_or_unknown_task(self) -> None:
        # Resuming a non-existent task ID should return an empty list and not crash
        resumed = resume_task(self.store, 999)
        assert resumed == []

        # Resuming a task that exists but has no feature_list should return an empty list
        task = self.store.create("Some other task")
        resumed_empty = resume_task(self.store, task.id)
        assert resumed_empty == []

    def test_progress_tools_basic_flow(self) -> None:
        task = self.store.create("Implement search")
        tools = {tool.name: tool for tool in build_progress_tools(self.store)}

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
        assert (self.root / "claude-progress.txt").exists()

    def test_start_run_dispatches_subtasks_and_can_resume(self) -> None:
        task = self.store.create("Implement orchestration")

        state = start_run(
            self.store,
            task.id,
            timeout_seconds=60,
            retry_limit=1,
            subtasks=["Write tests", "Update docs"],
        )
        resumed = resume_run(self.store, task.id)
        child_titles = [self.store.get(task_id).title for task_id in state.subtask_ids]

        assert state.status == "running"
        assert state.attempt == 1
        assert resumed == state
        assert child_titles == ["Write tests", "Update docs"]
        assert self.store.get(task.id).status == "claimed"

    def test_expire_stale_runs_releases_claimed_task(self) -> None:
        task = self.store.create("Long task")
        start_run(self.store, task.id, timeout_seconds=1, retry_limit=1)
        current = self.store.get(task.id)
        payload = dict(current.payload)
        orchestration = dict(payload["orchestration"])
        orchestration["lease_expires_at"] = "2000-01-01T00:00:00Z"
        payload["orchestration"] = orchestration
        self.store.update(task.id, payload=payload)

        expired = expire_stale_runs(self.store)

        assert len(expired) == 1
        assert expired[0].status == "timed_out"
        assert self.store.get(task.id).status == "pending"

    def test_retry_run_respects_retry_limit(self) -> None:
        task = self.store.create("Retry task")
        start_run(self.store, task.id, timeout_seconds=60, retry_limit=1)

        retried = retry_run(self.store, task.id)

        assert retried.attempt == 2
        with pytest.raises(ValueError):
            retry_run(self.store, task.id)

    def test_progress_tools_expose_orchestration_flow(self) -> None:
        task = self.store.create("Tool orchestration")
        tools = {tool.name: tool for tool in build_progress_tools(self.store)}

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
