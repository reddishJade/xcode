from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xcode.experimental.tasks import TaskStore
from xcode.experimental.progress import TaskProgress, build_progress_tools


class TestTaskProgress(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = TaskStore(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_resume_progress_basic(self) -> None:
        # 1. Create a task in the TaskStore
        task = self.store.create("Implement login feature")
        self.assertEqual(task.id, 1)

        # 2. Define the sub-tasks (feature list)
        checklist = [
            {"step": 1, "title": "Design schema", "status": "completed"},
            {"step": 2, "title": "Implement API", "status": "in_progress"},
            {"step": 3, "title": "Write unit tests", "status": "pending"},
        ]

        # 3. Save progress
        TaskProgress.save_progress(self.store, task.id, checklist)

        # 4. Assert SoT (payload in TaskStore) has been updated under lock protection
        updated_task = self.store.get(task.id)
        saved_checklist = updated_task.payload.get("feature_list")
        self.assertEqual(saved_checklist, checklist)

        # 5. Assert derived read-only view 'claude-progress.txt' was written to root
        progress_txt = self.root / "claude-progress.txt"
        self.assertTrue(progress_txt.exists())

        content = progress_txt.read_text(encoding="utf-8")
        self.assertIn("Progress: 33.3%", content)
        self.assertIn("Step 1: Design schema", content)
        self.assertIn("- [/] Step 2: Implement API", content)
        self.assertIn("- [ ] Step 3: Write unit tests", content)
        self.assertIn("Current Active Step:\n- Implement API", content)

        # 6. Assert resume_task fetches SoT checklist perfectly
        resumed_checklist = TaskProgress.resume_task(self.store, task.id)
        self.assertEqual(resumed_checklist, checklist)

    def test_resume_missing_or_unknown_task(self) -> None:
        # Resuming a non-existent task ID should return an empty list and not crash
        resumed = TaskProgress.resume_task(self.store, 999)
        self.assertEqual(resumed, [])

        # Resuming a task that exists but has no feature_list should return an empty list
        task = self.store.create("Some other task")
        resumed_empty = TaskProgress.resume_task(self.store, task.id)
        self.assertEqual(resumed_empty, [])

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
        self.assertEqual(saved, f"saved progress for task {task.id}")

        resumed = tools["resume_task_progress"].handler({"task_id": 1})
        self.assertIn('"title": "Index files"', resumed)
        self.assertIn('"status": "completed"', resumed)
        self.assertTrue((self.root / "claude-progress.txt").exists())


if __name__ == "__main__":
    unittest.main()
