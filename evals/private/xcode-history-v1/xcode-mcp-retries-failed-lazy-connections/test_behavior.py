"""Agent 不可见的 MCP lazy connection 恢复行为 oracle。"""

from unittest.mock import MagicMock, patch

import pytest

from xcode.harness.mcp.client import LazyClientRef, McpClient


def test_transient_failure_is_retried_in_the_same_request() -> None:
    first = MagicMock(spec=McpClient)
    first.start.side_effect = RuntimeError("temporary handshake failure")
    second = MagicMock(spec=McpClient)
    second.status = "connected"
    callback = MagicMock()
    reference = LazyClientRef(
        "fixture",
        {
            "command": "fixture-server",
            "args": ["--stdio"],
            "env": {"TOKEN": "value"},
            "timeout": 3.0,
        },
        tools_changed_callback=callback,
    )

    with patch(
        "xcode.harness.mcp.client.McpClient",
        side_effect=[first, second],
    ) as client_class:
        connected = reference.get_or_create()

    assert connected is second
    assert client_class.call_count == 2
    first.stop.assert_called_once_with()
    second.start.assert_called_once_with()
    second.set_tools_changed_callback.assert_called_once_with(callback)


def test_exhaustion_stops_clients_and_redacts_the_final_error() -> None:
    clients = [MagicMock(spec=McpClient), MagicMock(spec=McpClient)]
    clients[0].start.side_effect = RuntimeError("old failure")
    clients[1].start.side_effect = RuntimeError("Bearer private-token")
    reference = LazyClientRef(
        "fixture",
        {"command": "fixture-server"},
        max_connect_attempts=2,
    )

    with (
        patch("xcode.harness.mcp.client.McpClient", side_effect=clients),
        pytest.raises(RuntimeError, match="after 2 attempts") as error,
    ):
        reference.get_or_create()

    assert "private-token" not in str(error.value)
    assert "Bearer ****" in str(error.value)
    assert reference.last_error == "Bearer ****"
    for client in clients:
        client.stop.assert_called_once_with()


def test_empty_attempt_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LazyClientRef(
            "fixture",
            {"command": "fixture-server"},
            max_connect_attempts=0,
        )
