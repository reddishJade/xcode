from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xcode.harness.observability.permissions import (
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
)
from xcode.experimental.plugins import PluginManager
from xcode.harness.task_store import (
    TaskStore,
    resolve_task_dependencies,
    render_kanban_view,
)


class XcodeMcpOverrideSecurityTests(unittest.TestCase):
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
        result = engine.decide("read_file", '{"path": "a.txt"}')
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")

        # 2. Blacklisted tools should be denied
        result = engine.decide("bash", "echo hello")
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")

        # 3. Restricted directory arguments should be denied
        result = engine.decide("read_file", '{"path": "/private/secrets/key.txt"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")

        # 4. Any other non-whitelisted tool should ask
        result = engine.decide("write_file", '{"path": "a.txt"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "ask")

    def test_permission_engine_restricted_dirs_override_allowed(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission("read_file", "allow"),)
                ),
                restricted_dirs=("secrets",),
            )
        )
        result = engine.decide("read_file", '{"path": "secrets/key.txt"}')
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")

    def test_dynamic_plugin_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            # Write a dummy dynamic plugin python file
            plugin_code = (
                "from xcode.harness.skills import ToolSpec\n"
                "exposed_tools = [\n"
                "    ToolSpec(\n"
                "        name='plugin_calc',\n"
                "        description='Calculator',\n"
                "        input_hint='{}',\n"
                "        handler=lambda _data: '42',\n"
                "    )\n"
                "]\n"
                "exposed_hooks = {\n"
                "    'post_tool': lambda record: None\n"
                "}\n"
            )
            plugins_dir = project_root / ".local" / "plugins"
            plugins_dir.mkdir(parents=True, exist_ok=True)
            (plugins_dir / "calculator.py").write_text(plugin_code, encoding="utf-8")

            manager = PluginManager(project_root)
            data = manager.scan_and_load()

            self.assertEqual(len(data["tools"]), 1)
            self.assertEqual(data["tools"][0].name, "plugin_calc")
            self.assertEqual(data["tools"][0].handler({}), "42")
            self.assertIn("post_tool", data["hooks"])

    def test_task_dependency_topo_sort_and_kanban_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            # Create three tasks with dependencies: Task 3 depends on Task 2, Task 2 depends on Task 1
            store = TaskStore(project_root)
            t1 = store.create("Setup project", payload={})
            t2 = store.create("Write core codebase", payload={"blocked_by": t1.id})
            t3 = store.create("Write tests", payload={"blocked_by": t2.id})

            all_tasks = store.list()
            self.assertEqual(len(all_tasks), 3)

            # Resolve dependencies via topological sorting
            sorted_tasks = resolve_task_dependencies(all_tasks)
            self.assertEqual([t.id for t in sorted_tasks], [t1.id, t2.id, t3.id])

            # Verify circular dependency checking
            # Artificially modify t1 to block on t3 (creating t1 -> t3 -> t2 -> t1 loop)
            t1_record = store.get(t1.id)
            t1_record.payload["blocked_by"] = t3.id
            store._write(t1_record)

            with self.assertRaises(ValueError) as ctx:
                resolve_task_dependencies(store.list())
            self.assertIn("Circular dependency detected", str(ctx.exception))

            # Verify Kanban view renders correctly
            kanban_txt = render_kanban_view(all_tasks)
            self.assertIn("=== TASK KANBAN VIEW ===", kanban_txt)
            self.assertIn("[PENDING]", kanban_txt)
            self.assertIn("- #1: Setup project", kanban_txt)


if __name__ == "__main__":
    unittest.main()
