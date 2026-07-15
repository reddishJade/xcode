"""Agent 不可见的 MCP override 诊断行为 oracle。"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from xcode.harness.mcp.tools import build_mcp_tools


def _tool(name: str) -> dict[str, object]:
    return {
        "name": name,
        "description": "hidden test tool",
        "inputSchema": {"type": "object", "properties": {}},
    }


def test_unknown_exact_override_warns_but_wildcard_and_tools_remain_valid(
    tmp_path: Path,
    caplog,
) -> None:
    config_path = tmp_path / ".local/mcp_config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "demo": {
                        "command": "python",
                        "overrides": {
                            "missing": {"enabled": False},
                            "*": {"read_only": True},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client = MagicMock()
    client.protocol_version = "2025-11-25"
    client.server_info = {"name": "demo", "version": "1"}
    client.list_tools.return_value = [_tool("known")]

    with (
        patch("xcode.harness.mcp.client.McpClient", return_value=client),
        caplog.at_level(logging.WARNING),
    ):
        tools = build_mcp_tools(tmp_path)

    assert [tool.name for tool in tools] == ["mcp__demo__known"]
    assert tools[0].read_only
    assert "override references unknown tool 'missing'; ignored" in caplog.text
    assert "unknown tool '*'" not in caplog.text
