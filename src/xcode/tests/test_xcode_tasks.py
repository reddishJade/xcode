from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from xcode.experimental.tasks import CLAIMED, PENDING, TaskStore


class XcodeTaskStoreTests(unittest.TestCase):
    def test_create_writes_one_json_file_per_task_and_highwatermark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))

            first = store.create("first", {"kind": "test"})
            second = store.create("second")

            tasks_dir = Path(temp_dir) / ".local" / "tasks.json.d"
            self.assertEqual(first.id, 1)
            self.assertEqual(second.id, 2)
            self.assertEqual(first.status, PENDING)
            self.assertEqual(
                (tasks_dir / ".highwatermark").read_text(encoding="utf-8").strip(), "2"
            )
            self.assertTrue((tasks_dir / "1.json").exists())
            self.assertTrue((tasks_dir / "2.json").exists())

    def test_read_and_list_return_persisted_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("inspect", {"path": "core/app.py"})

            reloaded = TaskStore(Path(temp_dir)).get(created.id)
            listed = TaskStore(Path(temp_dir)).list()

            self.assertEqual(reloaded.title, "inspect")
            self.assertEqual(reloaded.payload, {"path": "core/app.py"})
            self.assertEqual([task.id for task in listed], [created.id])

    def test_update_rewrites_task_and_increments_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("old", {"a": 1})

            updated = store.update(
                created.id, title="new", status="done", payload={"b": 2}
            )
            reloaded = store.get(created.id)

            self.assertEqual(updated.version, 2)
            self.assertEqual(reloaded.title, "new")
            self.assertEqual(reloaded.status, "done")
            self.assertEqual(reloaded.payload, {"b": 2})

    def test_claim_moves_pending_task_to_claimed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("claim me")

            claimed = store.claim(created.id, "worker-a")
            second = store.claim(created.id, "worker-b")

            self.assertIsNotNone(claimed)
            assert claimed is not None
            self.assertEqual(claimed.status, CLAIMED)
            self.assertEqual(claimed.claimed_by, "worker-a")
            self.assertIsNone(second)

    def test_concurrent_claim_allows_only_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_id = TaskStore(root).create("race").id

            def claim(name: str):
                return TaskStore(root).claim(task_id, name)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(claim, ("worker-a", "worker-b")))

            winners = [result for result in results if result is not None]
            self.assertEqual(len(winners), 1)
            self.assertIn(
                TaskStore(root).get(task_id).claimed_by, {"worker-a", "worker-b"}
            )

    def test_task_tool_handlers(self) -> None:
        from xcode.experimental.tasks import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            tools = build_task_tools(store)

            create_tool = next(t for t in tools if t.name == "create_task")
            update_tool = next(t for t in tools if t.name == "update_task")
            list_tool = next(t for t in tools if t.name == "list_tasks")
            get_tool = next(t for t in tools if t.name == "get_task")

            # 1. Test create_task handler
            res_create = create_tool.handler(
                {"title": "Implement auth", "blocked_by": [10]}
            )
            self.assertIn("Created task #1: 'Implement auth'", res_create)

            # 2. Test get_task handler
            res_get = get_tool.handler({"id": 1})
            self.assertIn("Implement auth", res_get)

            # 3. Test list_tasks handler
            res_list = list_tool.handler({"view": "kanban"})
            self.assertIn("=== TASK KANBAN VIEW ===", res_list)
            self.assertIn("Implement auth", res_list)

            # 4. Test update_task handler
            res_update = update_tool.handler({"id": 1, "status": "completed"})
            self.assertIn("Updated task #1: status=completed", res_update)

            self.assertEqual(store.get(1).status, "completed")


if __name__ == "__main__":
    unittest.main()
