from __future__ import annotations

import tempfile
from pathlib import Path

from xcode.harness.observability.permissions import (
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.task_store import (
    TaskStore,
    resolve_task_dependencies,
    render_kanban_view,
)
import pytest


class TestXcodePermissionPipeline:
    def test_permission_engine_sandbox_equivalent(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    rules=(
                        StaticPermission("read_file", "allow"),
                        StaticPermission("bash", "deny"),
                    ),
                    global_default="ask",
                ),
                restricted_dirs=("/private/secrets", "C:/Windows"),
            )
        )

        # 1. Whitelisted tools should be allowed
        result = engine.decide("read_file", {"path": "a.txt"})
        assert not (result.blocked)
        assert result.decision == "allow"

        # 2. Blacklisted tools should be denied
        result = engine.decide("bash", {"command": "echo hello"})
        assert result.blocked
        assert result.decision == "deny"

        # 3. Restricted directory arguments should be denied
        result = engine.decide(
            "read_file",
            {"path": "/private/secrets/key.txt"},
        )
        assert result.blocked
        assert result.decision == "deny"

        # 4. Any other non-whitelisted tool should ask
        result = engine.decide("write_file", {"path": "a.txt"})
        assert result.blocked
        assert result.decision == "ask"

    def test_permission_engine_restricted_dirs_override_allowed(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission("read_file", "allow"),)
                ),
                restricted_dirs=("secrets",),
            )
        )
        result = engine.decide("read_file", {"path": "secrets/key.txt"})
        assert result.blocked
        assert result.decision == "deny"

    def test_task_dependency_topo_sort_and_kanban_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            # Create three tasks with dependencies: Task 3 depends on Task 2, Task 2 depends on Task 1
            store = TaskStore(project_root)
            t1 = store.create("Setup project", payload={})
            t2 = store.create("Write core codebase", payload={"blocked_by": [t1.id]})
            t3 = store.create("Write tests", payload={"blocked_by": [t2.id]})

            all_tasks = store.list()
            assert len(all_tasks) == 3

            # Resolve dependencies via topological sorting
            sorted_tasks = resolve_task_dependencies(all_tasks)
            assert [t.id for t in sorted_tasks] == [t1.id, t2.id, t3.id]

            # Verify circular dependency checking
            # Artificially modify t1 to block on t3 (creating t1 -> t3 -> t2 -> t1 loop)
            t1_record = store.get(t1.id)
            t1_record.payload["blocked_by"] = [t3.id]
            store._write(t1_record)

            with pytest.raises(ValueError) as exc_info:
                resolve_task_dependencies(store.list())
            assert "Circular dependency detected" in str(exc_info.value)

            # Verify Kanban view renders correctly
            kanban_txt = render_kanban_view(all_tasks)
            assert "=== TASK KANBAN VIEW ===" in kanban_txt
            assert "[PENDING]" in kanban_txt
            assert "- #1: Setup project" in kanban_txt


if __name__ == "__main__":
    pytest.main()
