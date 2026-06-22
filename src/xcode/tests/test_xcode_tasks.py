from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from xcode.harness.task_store import CLAIMED, PENDING, TaskStore
import pytest
class XcodeTaskStoreTests:
    def test_create_writes_one_json_file_per_task_and_highwatermark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))

            first = store.create("first", {"kind": "test"})
            second = store.create("second")

            tasks_dir = Path(temp_dir) / ".local" / "tasks.json.d"
            assert first.id == 1
            assert second.id == 2
            assert first.status == PENDING
            assert (tasks_dir / ".highwatermark").read_text(encoding="utf-8").strip() == "2"
            assert (tasks_dir / "1.json").exists()
            assert (tasks_dir / "2.json").exists()

    def test_read_and_list_return_persisted_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("inspect", {"path": "core/app.py"})

            reloaded = TaskStore(Path(temp_dir)).get(created.id)
            listed = TaskStore(Path(temp_dir)).list()

            assert reloaded.title == "inspect"
            assert reloaded.payload == {"path": "core/app.py"}
            assert [task.id for task in listed] == [created.id]

    def test_update_rewrites_task_and_increments_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("old", {"a": 1})

            updated = store.update(
                created.id, title="new", status="done", payload={"b": 2}
            )
            reloaded = store.get(created.id)

            assert updated.version == 2
            assert reloaded.title == "new"
            assert reloaded.status == "done"
            assert reloaded.payload == {"b": 2}

    def test_claim_moves_pending_task_to_claimed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("claim me")

            claimed = store.claim(created.id, "worker-a")
            second = store.claim(created.id, "worker-b")

            assert claimed is not None
            assert claimed is not None
            assert claimed.status == CLAIMED
            assert claimed.claimed_by == "worker-a"
            assert second is None

    def test_concurrent_claim_allows_only_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_id = TaskStore(root).create("race").id

            def claim(name: str):
                return TaskStore(root).claim(task_id, name)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(claim, ("worker-a", "worker-b")))

            winners = [result for result in results if result is not None]
            assert len(winners) == 1
            assert TaskStore(root).get(task_id).claimed_by in {"worker-a", "worker-b"}

    def test_task_tool_handlers(self) -> None:
        from xcode.harness.task_store import build_task_tools
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
            assert "Created task #1: 'Implement auth'" in res_create

            # 2. Test get_task handler
            res_get = get_tool.handler({"id": 1})
            assert "Implement auth" in res_get

            # 3. Test list_tasks handler
            res_list = list_tool.handler({"view": "kanban"})
            assert "=== TASK KANBAN VIEW ===" in res_list
            assert "Implement auth" in res_list

            # 4. Test update_task handler
            res_update = update_tool.handler({"id": 1, "status": "completed"})
            assert "Updated task #1: status=completed" in res_update

            assert store.get(1).status == "completed"

if __name__ == "__main__":
    pytest.main()
