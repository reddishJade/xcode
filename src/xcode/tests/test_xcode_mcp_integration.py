from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from xcode.harness.mcp.tools import (
    McpRuntimeRegistry,
    build_mcp_tools,
    compute_config_hash,
)
from xcode.harness.skills import ToolSpec


class XcodeMcpIntegrationTests:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)

    def teardown_method(self, method) -> None:
        shutil.rmtree(self.temp_dir)

    def test_compute_config_hash(self) -> None:
        config1 = {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {"DEBUG": "1"},
        }
        config2 = {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {"DEBUG": "1"},
        }
        config3 = {
            "command": "python",
            "args": ["-m", "mcp_server", "--extra"],
            "env": {"DEBUG": "1"},
        }
        assert compute_config_hash(config1) == compute_config_hash(config2)
        assert compute_config_hash(config1) != compute_config_hash(config3)

    @patch("xcode.harness.mcp.client.McpClient")
    def test_build_mcp_tools_cache_hit_and_miss(
        self, mock_client_class: MagicMock
    ) -> None:
        # Define mock client behavior
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.protocol_version = "2025-11-25"
        mock_client.server_info = {"name": "fetcher", "version": "1.0.0"}
        mock_client.list_tools.return_value = [
            {
                "name": "read_data",
                "description": "Read remote data",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            }
        ]

        # 1. Prepare config files — no overrides (Step 9 removes them)
        mcp_config = {
            "mcpServers": {
                "fetcher": {
                    "command": "python",
                    "args": ["fetch_server.py"],
                    "env": {},
                }
            }
        }
        local_dir = self.project_root / ".local"
        local_dir.mkdir(parents=True)
        (local_dir / "mcp_config.json").write_text(
            json.dumps(mcp_config), encoding="utf-8"
        )

        # 2. Trigger cache miss: should query client
        tools = build_mcp_tools(self.project_root)
        assert len(tools) == 1
        assert tools[0].name == "mcp__fetcher__read_data"
        assert tools[0].description == "Read remote data [mcp: fetcher]"
        assert tools[0].group == "mcp"

        # Verify client was instantiated, started, queried, and stopped
        mock_client_class.assert_called_once_with(
            ["python", "fetch_server.py"], {}, timeout=None, workspace_roots=()
        )
        mock_client.start.assert_called_once()
        mock_client.list_tools.assert_called_once()
        mock_client.stop.assert_called_once()

        # Cache file should be generated
        cache_path = local_dir / "mcp_cache.json"
        assert cache_path.exists()
        cache_content = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "fetcher" in cache_content["servers"]
        assert cache_content["servers"]["fetcher"]["protocol_version"] == "2025-11-25"
        assert cache_content["servers"]["fetcher"]["server_info"] == {
            "name": "fetcher",
            "version": "1.0.0",
        }

        # Reset mocks
        mock_client_class.reset_mock()
        mock_client.reset_mock()

        # 3. Trigger cache hit: should NOT query client, just read cache
        tools2 = build_mcp_tools(self.project_root)
        assert len(tools2) == 1
        assert tools2[0].name == "mcp__fetcher__read_data"
        mock_client_class.assert_not_called()

        # 4. Trigger config change -> cache miss again
        mcp_config["mcpServers"]["fetcher"]["args"] = ["fetch_server.py", "--verbose"]
        (local_dir / "mcp_config.json").write_text(
            json.dumps(mcp_config), encoding="utf-8"
        )

        build_mcp_tools(self.project_root)
        mock_client_class.assert_called_once_with(
            ["python", "fetch_server.py", "--verbose"],
            {},
            timeout=None,
            workspace_roots=(),
        )

    @patch("xcode.harness.mcp.client.McpClient")
    def test_build_mcp_tools_lazy_initialization(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.protocol_version = "2025-11-25"
        mock_client.server_info = {"name": "db-editor", "version": "1.0.0"}
        mock_client.list_tools.return_value = [
            {
                "name": "edit_record",
                "description": "Edit a database record",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "value": {"type": "string"},
                    },
                    "required": ["id", "value"],
                },
            }
        ]
        mock_client.call_tool.return_value = {
            "content": [{"type": "text", "text": "Success"}]
        }

        mcp_config = {
            "mcpServers": {
                "db_editor": {
                    "command": "python",
                    "args": ["db_server.py"],
                }
            }
        }
        local_dir = self.project_root / ".local"
        local_dir.mkdir(parents=True)
        (local_dir / "mcp_config.json").write_text(
            json.dumps(mcp_config), encoding="utf-8"
        )

        # build_mcp_tools will trigger handshake to get schema first, then close client
        tools = build_mcp_tools(self.project_root)
        assert len(tools) == 1
        assert tools[0].name == "mcp__db_editor__edit_record"
        tool = tools[0]
        assert tool.name == "mcp__db_editor__edit_record"
        assert not (tool.read_only)

        # Reset mocks to verify lazy loading on execution
        mock_client_class.reset_mock()
        mock_client.reset_mock()

        # When we build tools, McpClient should NOT start a persistent connection
        # The connection should only be created and started when we invoke the handler
        mock_client_class.assert_not_called()

        # Invoke handler
        response = tool.handler({"id": 42, "value": "new_val"})
        assert response == "Success"

        # Now connection should be created and started, and call_tool invoked
        mock_client_class.assert_called_once_with(
            ["python", "db_server.py"], None, timeout=None, workspace_roots=()
        )
        mock_client.start.assert_called_once()
        mock_client.call_tool.assert_called_once_with(
            "edit_record",
            {"id": 42, "value": "new_val"},
            timeout=None,
            progress_callback=mock_client.call_tool.call_args.kwargs[
                "progress_callback"
            ],
            cancel_event=None,
        )

    @patch("xcode.harness.mcp.client.McpClient")
    def test_tools_changed_refreshes_cache_and_runtime_registry(
        self,
        mock_client_class: MagicMock,
    ) -> None:
        """listChanged 会同步 schema cache 和运行时工具快照。"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.protocol_version = "2025-11-25"
        mock_client.server_info = {"name": "dynamic", "version": "1.0.0"}
        mock_client.list_tools.return_value = [
            {
                "name": "old_tool",
                "description": "Old schema",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        mock_client.call_tool.return_value = {
            "content": [{"type": "text", "text": "ok"}]
        }
        local_dir = self.project_root / ".local"
        local_dir.mkdir(parents=True)
        (local_dir / "mcp_config.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "dynamic": {
                            "command": "python",
                            "args": ["dynamic_server.py"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        runtime_registry = McpRuntimeRegistry()
        published: list[tuple[ToolSpec, ...]] = []
        runtime_registry.subscribe(published.append)

        tools = build_mcp_tools(self.project_root, runtime_registry)
        tools[0].handler({})
        callback = mock_client.set_tools_changed_callback.call_args.args[0]
        mock_client.list_tools.return_value = [
            {
                "name": "new_tool",
                "description": "New schema",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ]

        callback(mock_client)

        assert [tool.name for tool in published[-1]] == ["mcp__dynamic__new_tool"]
        assert published[-1][0].schema == {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        cache = json.loads((local_dir / "mcp_cache.json").read_text(encoding="utf-8"))
        assert cache["servers"]["dynamic"]["tools"][0]["name"] == "new_tool"
        runtime_registry.close()
        mock_client.stop.assert_called()

    @patch("xcode.harness.mcp.client.McpClient")
    def test_legacy_cache_without_negotiation_metadata_is_refreshed(
        self,
        mock_client_class: MagicMock,
    ) -> None:
        """仅有 config hash 的旧缓存不能绕过协议和身份协商。"""
        mock_client = MagicMock()
        mock_client.protocol_version = "2025-11-25"
        mock_client.server_info = {"name": "refresh", "version": "2.0.0"}
        mock_client.list_tools.return_value = [
            {
                "name": "fresh_tool",
                "description": "Fresh",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        mock_client_class.return_value = mock_client
        raw_config = {"command": "python", "args": ["server.py"]}
        local_dir = self.project_root / ".local"
        local_dir.mkdir(parents=True)
        (local_dir / "mcp_config.json").write_text(
            json.dumps({"mcpServers": {"refresh": raw_config}}),
            encoding="utf-8",
        )
        (local_dir / "mcp_cache.json").write_text(
            json.dumps(
                {
                    "servers": {
                        "refresh": {
                            "config_hash": compute_config_hash(raw_config),
                            "tools": [
                                {
                                    "name": "stale_tool",
                                    "inputSchema": {"type": "object"},
                                }
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        tools = build_mcp_tools(self.project_root)

        assert [tool.name for tool in tools] == ["mcp__refresh__fresh_tool"]
        mock_client.list_tools.assert_called_once()
