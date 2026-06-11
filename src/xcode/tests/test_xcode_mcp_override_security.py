from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from xcode.harness.observability.permissions import (
    SettingsSandboxPermissionPolicy,
)
from xcode.experimental.plugins import PluginManager
from xcode.experimental.mcp import build_mcp_tools
from xcode.harness.task_store import (
    TaskStore,
    resolve_task_dependencies,
    render_kanban_view,
)


class XcodeMcpOverrideSecurityTests(unittest.TestCase):
    def test_mcp_risk_override_and_default_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            mcp_config = {
                "mcpServers": {
                    "test-server": {
                        "command": "node",
                        "args": ["a.js"],
                        "defer_loading": True,
                        "overrides": {"high_risk_tool": "low"},
                    }
                }
            }
            (project_root / "mcp_config.json").write_text(
                json.dumps(mcp_config, indent=2), encoding="utf-8"
            )

            mcp_cache = {
                "servers": {
                    "test-server": {
                        "config_hash": "dummy_hash",
                        "tools": [
                            {
                                "name": "high_risk_tool",
                                "description": "Runs high risk cmd",
                            },
                            {
                                "name": "default_low_tool",
                                "description": "Reads low info",
                            },
                        ],
                    }
                }
            }
            cache_path = project_root / ".local" / "mcp_cache.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(mcp_cache, indent=2), encoding="utf-8")

            # Monkeypatch compute_config_hash to return dummy_hash
            from xcode.experimental import mcp

            orig_hash = mcp.compute_config_hash
            try:
                mcp.compute_config_hash = lambda server_config: "dummy_hash"
                tools = build_mcp_tools(project_root)
            finally:
                mcp.compute_config_hash = orig_hash

            # Filter returned specs
            high_tool = next(
                t for t in tools if t.name == "mcp__test-server__high_risk_tool"
            )
            low_tool = next(
                t for t in tools if t.name == "mcp__test-server__default_low_tool"
            )

            self.assertEqual(high_tool.risk, "low")
            self.assertEqual(low_tool.risk, "high")

    def test_settings_sandbox_permissions_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            # Setup settings.json permissions
            settings = {
                "allowedTools": ["read_file"],
                "deniedTools": ["bash"],
                "restrictedDirs": ["/private/secrets", "C:/Windows"],
            }
            settings_path = project_root / ".local" / "settings.json"
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

            policy = SettingsSandboxPermissionPolicy(settings_path)

            # 1. Whitelisted tools should be allowed
            self.assertEqual(policy.decide("read_file", '{"path": "a.txt"}'), "allow")

            # 2. Blacklisted tools should be denied
            self.assertEqual(policy.decide("bash", "echo hello"), "deny")

            # 3. Restricted directory arguments should be denied
            self.assertEqual(
                policy.decide("read_file", '{"path": "/private/secrets/key.txt"}'),
                "deny",
            )

            # 4. Any other non-whitelisted tool should ask
            self.assertEqual(policy.decide("write_file", '{"path": "a.txt"}'), "ask")

    def test_settings_sandbox_restricted_dirs_override_allowed_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            settings_path = project_root / ".local" / "settings.json"
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(
                json.dumps(
                    {
                        "security": {
                            "allow_tools": ["read_file"],
                            "restricted_dirs": ["secrets"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            policy = SettingsSandboxPermissionPolicy(settings_path)

            self.assertEqual(
                policy.decide("read_file", '{"path": "secrets/key.txt"}'),
                "deny",
            )

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
                "        risk='low'\n"
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
