"""官方 MCP SDK 客户端适配层测试。"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xcode.harness.mcp.client import (
    LATEST_PROTOCOL_VERSION,
    MAX_LAZY_CONNECT_ATTEMPTS,
    LazyClientRef,
    McpClient,
)

SERVER_CODE = r"""
import json
import sys

TOOLS = [
    {
        "name": "mock_read_tool",
        "description": "Read mock data",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]

def read_message():
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8"))

def write_message(message):
    sys.stdout.buffer.write((json.dumps(message) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

def main():
    while True:
        request = read_message()
        if request is None:
            return
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            continue
        if method == "initialize":
            write_message({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "mock-server", "version": "1.0.0"},
                    "instructions": "Use tools carefully.",
                },
            })
        elif method == "tools/list":
            write_message({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": TOOLS},
            })
        elif method == "tools/call":
            path = request.get("params", {}).get("arguments", {}).get(
                "path", "default"
            )
            write_message({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {"type": "text", "text": f"Content of {path}"}
                    ],
                    "isError": False,
                },
            })

if __name__ == "__main__":
    main()
"""


def _server_command(temp_dir: str, code: str = SERVER_CODE) -> list[str]:
    server_file = Path(temp_dir) / "mcp_server.py"
    server_file.write_text(code, encoding="utf-8")
    return [sys.executable, "-u", str(server_file)]


class TestMcpSdkClient:
    """验证同步适配层保留 Xcode 所需的公开行为。"""

    def test_handshake_list_call_and_stop(self) -> None:
        """SDK 完成握手、工具发现、调用和关闭。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = McpClient(_server_command(temp_dir))
            client.start()

            assert client.status == "connected"
            assert client.protocol_version == "2024-11-05"
            assert client.server_info == {
                "name": "mock-server",
                "version": "1.0.0",
            }
            assert client.instructions == "Use tools carefully."
            assert client.has_server_capability("tools")

            tools = client.list_tools()
            assert [tool["name"] for tool in tools] == ["mock_read_tool"]

            result = client.call_tool("mock_read_tool", {"path": "test.txt"})
            assert result["content"][0]["text"] == "Content of test.txt"

            client.stop()
            assert client.status == "disabled"

    def test_initialize_uses_sdk_protocol_version(self) -> None:
        """适配层公开的最新版本来自官方 SDK。"""
        assert LATEST_PROTOCOL_VERSION == "2025-11-25"

    def test_rejects_tools_without_negotiated_capability(self) -> None:
        """未协商 tools 时不允许调用工具方法。"""
        no_tools_code = SERVER_CODE.replace(
            '"capabilities": {"tools": {"listChanged": True}}',
            '"capabilities": {}',
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            client = McpClient(_server_command(temp_dir, no_tools_code))
            client.start()

            with pytest.raises(RuntimeError, match="did not negotiate 'tools'"):
                client.list_tools()

            client.stop()

    def test_rejects_calls_after_stop(self) -> None:
        """关闭后的 client 不可继续复用 session。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = McpClient(_server_command(temp_dir))
            client.start()
            client.stop()

            with pytest.raises(RuntimeError, match="not connected"):
                client.call_tool("mock_read_tool", {})

    def test_tools_changed_notification_runs_callback_off_sdk_loop(self) -> None:
        """工具变化回调可同步调用 client，而不会阻塞 SDK receive loop。"""
        callback_done = threading.Event()
        callback_tools: list[str] = []

        def refresh(client: McpClient) -> None:
            callback_tools.extend(tool["name"] for tool in client.list_tools())
            callback_done.set()

        with tempfile.TemporaryDirectory() as temp_dir:
            client = McpClient(
                _server_command(temp_dir),
                tools_changed_callback=refresh,
            )
            client.start()
            client._schedule_tools_refresh()

            assert callback_done.wait(timeout=2.0)
            assert callback_tools == ["mock_read_tool"]
            client.stop()

    def test_lazy_client_ref_retries_and_reuses_connected_client(self) -> None:
        """首次连接失败后有限重试，并复用成功连接。"""
        callback = MagicMock()
        first_client = MagicMock(spec=McpClient)
        first_client.start.side_effect = RuntimeError("first failure")
        second_client = MagicMock(spec=McpClient)
        second_client.status = "connected"
        ref = LazyClientRef(
            "fixture",
            {
                "command": "python",
                "args": ["server.py"],
                "env": {"TOKEN": "value"},
                "timeout": 4.0,
            },
            tools_changed_callback=callback,
        )

        with patch(
            "xcode.harness.mcp.client.McpClient",
            side_effect=[first_client, second_client],
        ) as client_class:
            connected = ref.get_or_create()
            reused = ref.get_or_create()

        assert connected is second_client
        assert reused is second_client
        assert ref.last_error is None
        assert client_class.call_count == MAX_LAZY_CONNECT_ATTEMPTS
        client_class.assert_any_call(
            ["python", "server.py"],
            {"TOKEN": "value"},
            timeout=4.0,
        )
        first_client.stop.assert_called_once_with()
        second_client.set_tools_changed_callback.assert_called_once_with(callback)

    def test_lazy_client_ref_redacts_retry_error(self) -> None:
        """重试耗尽后保留脱敏错误。"""
        clients = [MagicMock(spec=McpClient) for _ in range(2)]
        clients[0].start.side_effect = RuntimeError("old failure")
        clients[1].start.side_effect = RuntimeError("Bearer secret-token")
        ref = LazyClientRef(
            "fixture",
            {"command": "server"},
            max_connect_attempts=len(clients),
        )

        with (
            patch("xcode.harness.mcp.client.McpClient", side_effect=clients),
            pytest.raises(RuntimeError, match="after 2 attempts"),
        ):
            ref.get_or_create()

        assert ref.last_error == "Bearer ****"
        for client in clients:
            client.stop.assert_called_once_with()

    def test_lazy_client_ref_rejects_empty_attempt_budget(self) -> None:
        """连接尝试次数必须至少为一。"""
        with pytest.raises(ValueError, match="at least 1"):
            LazyClientRef(
                "fixture",
                {"command": "server"},
                max_connect_attempts=0,
            )
