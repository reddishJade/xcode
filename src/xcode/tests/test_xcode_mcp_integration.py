from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from xcode.harness.mcp.tools import (
    build_mcp_tools,
    compute_config_hash,
)


class XcodeMcpIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)

    def tearDown(self) -> None:
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
        self.assertEqual(compute_config_hash(config1), compute_config_hash(config2))
        self.assertNotEqual(compute_config_hash(config1), compute_config_hash(config3))

    @patch("xcode.harness.mcp.client.McpClient")
    def test_build_mcp_tools_cache_hit_and_miss(
        self, mock_client_class: MagicMock
    ) -> None:
        # Define mock client behavior
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
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
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "mcp__fetcher__read_data")
        self.assertEqual(tools[0].description, "Read remote data [mcp: fetcher]")
        self.assertEqual(tools[0].group, "mcp")

        # Verify client was instantiated, started, queried, and stopped
        mock_client_class.assert_called_once_with(
            ["python", "fetch_server.py"], {}, timeout=None
        )
        mock_client.start.assert_called_once()
        mock_client.list_tools.assert_called_once()
        mock_client.stop.assert_called_once()

        # Cache file should be generated
        cache_path = local_dir / "mcp_cache.json"
        self.assertTrue(cache_path.exists())
        cache_content = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertIn("fetcher", cache_content["servers"])

        # Reset mocks
        mock_client_class.reset_mock()
        mock_client.reset_mock()

        # 3. Trigger cache hit: should NOT query client, just read cache
        tools2 = build_mcp_tools(self.project_root)
        self.assertEqual(len(tools2), 1)
        self.assertEqual(tools2[0].name, "mcp__fetcher__read_data")
        mock_client_class.assert_not_called()

        # 4. Trigger config change -> cache miss again
        mcp_config["mcpServers"]["fetcher"]["args"] = ["fetch_server.py", "--verbose"]
        (local_dir / "mcp_config.json").write_text(
            json.dumps(mcp_config), encoding="utf-8"
        )

        build_mcp_tools(self.project_root)
        mock_client_class.assert_called_once_with(
            ["python", "fetch_server.py", "--verbose"], {}, timeout=None
        )

    @patch("xcode.harness.mcp.client.McpClient")
    def test_build_mcp_tools_lazy_initialization(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
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
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "mcp__db_editor__edit_record")
        tool = tools[0]
        self.assertEqual(tool.name, "mcp__db_editor__edit_record")
        self.assertFalse(tool.read_only)

        # Reset mocks to verify lazy loading on execution
        mock_client_class.reset_mock()
        mock_client.reset_mock()

        # When we build tools, McpClient should NOT start a persistent connection
        # The connection should only be created and started when we invoke the handler
        mock_client_class.assert_not_called()

        # Invoke handler
        response = tool.handler({"id": 42, "value": "new_val"})
        self.assertEqual(response, "Success")

        # Now connection should be created and started, and call_tool invoked
        mock_client_class.assert_called_once_with(
            ["python", "db_server.py"], None, timeout=None
        )
        mock_client.start.assert_called_once()
        mock_client.call_tool.assert_called_once_with(
            "edit_record", {"id": 42, "value": "new_val"}, timeout=None
        )
