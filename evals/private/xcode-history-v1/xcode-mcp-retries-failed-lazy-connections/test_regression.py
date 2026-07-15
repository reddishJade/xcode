"""Agent 不可见的 MCP lazy connection 稳定回归。"""

from unittest.mock import MagicMock, patch

from xcode.harness.mcp.client import LazyClientRef, McpClient


def test_connected_client_is_reused_and_stop_clears_it() -> None:
    client = MagicMock(spec=McpClient)
    client.status = "connected"
    reference = LazyClientRef("fixture", {"command": "fixture-server"})

    with patch("xcode.harness.mcp.client.McpClient", return_value=client) as factory:
        assert reference.get_or_create() is client
        assert reference.get_or_create() is client

    factory.assert_called_once_with(["fixture-server"], None, timeout=None)
    client.start.assert_called_once_with()
    reference.stop()
    client.stop.assert_called_once_with()
    assert reference.client is None
