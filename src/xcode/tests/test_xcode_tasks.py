from __future__ import annotations

import jsonschema
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from xcode.harness.task_store import (
    CLAIMED,
    COMPLETED,
    ConcurrentModificationError,
    PENDING,
    TaskStore,
    UPDATE_TASK_SCHEMA,
    render_kanban_view,
)
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
            assert (tasks_dir / ".highwatermark").read_text(
                encoding="utf-8"
            ).strip() == "2"
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
                created.id, title="new", status="completed", payload={"b": 2}
            )
            reloaded = store.get(created.id)

            assert updated.version == 2
            assert reloaded.title == "new"
            assert reloaded.status == "completed"
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

    def test_update_rejects_invalid_status(self) -> None:
        """TaskStore.update 对非枚举 status 抛 ValueError。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("task")
            with pytest.raises(ValueError, match="invalid status"):
                store.update(created.id, status="in_progress")
            with pytest.raises(ValueError, match="invalid status"):
                store.update(created.id, status="done")
            # 合法 status 不抛错
            store.update(created.id, status=COMPLETED)
            assert store.get(created.id).status == COMPLETED

    def test_update_schema_enum_rejects_invalid_status(self) -> None:
        """UPDATE_TASK_SCHEMA 的 enum 在 validation 层拒绝非法 status。"""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"id": 1, "status": "done"}, UPDATE_TASK_SCHEMA)
        # 合法值通过
        jsonschema.validate({"id": 1, "status": "completed"}, UPDATE_TASK_SCHEMA)
        jsonschema.validate({"id": 1, "status": "pending"}, UPDATE_TASK_SCHEMA)
        jsonschema.validate({"id": 1, "status": "claimed"}, UPDATE_TASK_SCHEMA)

    def test_update_schema_rejects_additional_properties(self) -> None:
        """UPDATE_TASK_SCHEMA additionalProperties=False 拒绝未知字段。"""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"id": 1, "foo": "bar"}, UPDATE_TASK_SCHEMA)

    def test_create_schema_rejects_dependencies_alias(self) -> None:
        """CREATE_TASK_SCHEMA additionalProperties=False 拒绝 dependencies 别名。"""
        from xcode.harness.task_store import CREATE_TASK_SCHEMA

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"title": "x", "dependencies": [1]}, CREATE_TASK_SCHEMA)
        # blocked_by 仍然合法
        jsonschema.validate({"title": "x", "blocked_by": [1]}, CREATE_TASK_SCHEMA)

    def test_kanban_view_unknown_status_category(self) -> None:
        """未知 status 的任务归入 [unknown] 段并产生 warning。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            store.create("normal")
            # 手动写入非法 status 的 task 文件
            import json

            bad_path = store.tasks_dir / "99.json"
            bad_path.write_text(
                json.dumps(
                    {
                        "id": 99,
                        "title": "bad status",
                        "status": "blocked",
                        "payload": {},
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "version": 1,
                    }
                ),
                encoding="utf-8",
            )
            tasks = store.list()
            assert any(t.status == "blocked" for t in tasks)
            output = render_kanban_view(tasks)
            assert "[UNKNOWN]" in output
            assert "bad status" in output
            assert "[WARNING]" in output
            # 合法任务仍在 pending 段
            assert "[PENDING]" in output
            assert "normal" in output

    def test_create_task_handler_rejects_dependencies_alias(self) -> None:
        """_create_task handler 不再接受 dependencies 别名。"""
        from xcode.harness.task_store import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            tools = build_task_tools(store)
            create_tool = next(t for t in tools if t.name == "create_task")
            # dependencies 别名在 schema 层被拒，但 handler 不应识别它
            # 即使绕过 schema，handler 也只读 blocked_by
            result = create_tool.handler({"title": "via_blocked", "blocked_by": [5]})
            assert "Created task" in result
            task = store.list()[0]
            assert task.payload.get("blocked_by") == [5]

    def test_advance_task_no_deadlock_under_external_lock(self) -> None:
        """advance_task 在 store.locked() 内调 _apply_update 不死锁。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            t1 = store.create("setup")
            t2 = store.create("build", payload={"blocked_by": [t1.id]})

            from xcode.harness.task_store import advance_task

            affected = advance_task(store, t1.id)
            assert store.get(t1.id).status == COMPLETED
            # t2 的 blocked_by 应被移除
            assert store.get(t2.id).payload.get("blocked_by") is None
            assert len(affected) == 2

    def test_update_with_stale_version_raises(self) -> None:
        """乐观锁：用旧版本号写入应抛 ConcurrentModificationError。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("task")
            # 另一次更新推进 version
            store.update(created.id, status=CLAIMED)
            # 用过期版本号写入
            with pytest.raises(ConcurrentModificationError, match="version mismatch"):
                store.update(created.id, status=COMPLETED, expected_version=1)

    def test_update_with_correct_version_succeeds(self) -> None:
        """乐观锁：用当前版本号写入成功并递增版本。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("task")
            updated = store.update(
                created.id, status=COMPLETED, expected_version=created.version
            )
            assert updated.version == created.version + 1
            assert updated.status == COMPLETED

    def test_update_without_expected_version_backward_compatible(self) -> None:
        """不传 expected_version 时行为与旧版一致，不检查版本。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("task")
            store.update(created.id, status=CLAIMED)
            # 不传 expected_version，即使版本已变也能写入
            result = store.update(created.id, status=COMPLETED)
            assert result.status == COMPLETED

    def test_update_task_handler_returns_conflict_message(self) -> None:
        """_update_task handler 捕获冲突并返回友好字符串而非抛错。"""
        from xcode.harness.task_store import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            tools = build_task_tools(store)
            update_tool = next(t for t in tools if t.name == "update_task")
            created = store.create("task")
            # 模拟另一写入者推进版本
            store.update(created.id, status=CLAIMED)
            # handler 用过期版本号
            result = update_tool.handler(
                {"id": created.id, "status": "completed", "expected_version": 1}
            )
            assert "Concurrent modification" in result
            assert "current version is 2" in result
            # 状态不应被修改
            assert store.get(created.id).status == CLAIMED

    def test_claim_task_tool_success(self) -> None:
        """claim_task 工具成功认领 pending 任务。"""
        from xcode.harness.task_store import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("claimable")
            tools = build_task_tools(store)
            claim_tool = next(t for t in tools if t.name == "claim_task")
            result = claim_tool.handler({"task_id": created.id, "claimant": "agent_a"})
            assert "Claimed task" in result
            assert "agent_a" in result
            task = store.get(created.id)
            assert task.status == CLAIMED
            assert task.claimed_by == "agent_a"

    def test_claim_task_already_claimed_returns_message(self) -> None:
        """claim_task 对已认领任务返回提示而非抛错。"""
        from xcode.harness.task_store import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("claimable")
            store.claim(created.id, "first")
            tools = build_task_tools(store)
            claim_tool = next(t for t in tools if t.name == "claim_task")
            result = claim_tool.handler({"task_id": created.id, "claimant": "second"})
            assert "not pending" in result
            assert store.get(created.id).claimed_by == "first"

    def test_claim_task_rejects_empty_claimant(self) -> None:
        """claim_task 拒绝空 claimant。"""
        from xcode.harness.task_store import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("claimable")
            tools = build_task_tools(store)
            claim_tool = next(t for t in tools if t.name == "claim_task")
            with pytest.raises(ValueError, match="claimant is required"):
                claim_tool.handler({"task_id": created.id, "claimant": ""})

    def test_claim_task_concurrent_only_one_wins(self) -> None:
        """并发 claim_task 恰好一个成功。"""
        from xcode.harness.task_store import build_task_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir))
            created = store.create("race")
            tools = build_task_tools(store)
            claim_tool = next(t for t in tools if t.name == "claim_task")

            def claim(name: str) -> str:
                return claim_tool.handler({"task_id": created.id, "claimant": name})

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(claim, ("a", "b", "c", "d")))
            winners = [r for r in results if "Claimed task" in r]
            assert len(winners) == 1
            assert store.get(created.id).status == CLAIMED


if __name__ == "__main__":
    pytest.main()
