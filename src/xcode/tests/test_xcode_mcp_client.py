"""MCP 客户端握手、协商和工具调用测试。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from xcode.harness.mcp.client import (
    KILL_GRACE_SECONDS,
    LATEST_PROTOCOL_VERSION,
    MAX_LAZY_CONNECT_ATTEMPTS,
    MAX_TOOL_LIST_PAGES,
    LazyClientRef,
    McpClient,
    SHUTDOWN_GRACE_SECONDS,
    TERMINATE_GRACE_SECONDS,
)
import pytest
from xcode.tests._helpers import assert_logs

MOCK_SERVER_CODE = r"""
import sys
import json

def read_message():
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    text = line.decode("utf-8").strip()
    if not text:
        return None
    return json.loads(text)

def write_message(message):
    sys.stdout.buffer.write((json.dumps(message) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

def main():
    while True:
        req = read_message()
        if req is None:
            break
        try:
            req_id = req.get("id")
            method = req.get("method")
            if req_id is not None:
                if method == "initialize":
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "mock-server", "version": "1.0.0"},
                            "instructions": "Use tools carefully."
                        }
                    }
                elif method == "tools/list":
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "mock_read_tool",
                                    "description": "Read mock data",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "path": {"type": "string"}
                                        },
                                        "required": ["path"]
                                    }
                                }
                            ]
                        }
                    }
                elif method == "tools/call":
                    args = req.get("params", {}).get("arguments", {})
                    path = args.get("path", "default")
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"Content of {path}"}
                            ]
                        }
                    }
                else:
                    res = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method {method} not found"}
                    }
                write_message(res)
        except Exception:
            pass

if __name__ == "__main__":
    main()
"""

INTERLEAVED_SERVER_CODE = r"""
import sys
import json
def read_message():
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    text = line.decode("utf-8").strip()
    return json.loads(text) if text else None

def write_message(message):
    sys.stdout.buffer.write((json.dumps(message) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

def main():
    while True:
        req = read_message()
        if req is None:
            break
        req_id = req.get("id")
        method = req.get("method")
        if req_id is None:
            continue
        if method == "initialize":
            write_message({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "interleaved", "version": "1.0.0"}
                }
            })
            continue
        if method != "tools/list":
            continue

        write_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {}
        })
        ping_response = read_message()

        write_message({
            "jsonrpc": "2.0",
            "id": "server-unknown",
            "method": "custom/request",
            "params": {}
        })
        unknown_response = read_message()

        write_message({
            "jsonrpc": "2.0",
            "method": "notifications/custom",
            "params": {}
        })

        tools = []
        if ping_response == {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {}
        }:
            tools.append({"name": "ping_ok", "inputSchema": {"type": "object"}})
        if (
            unknown_response.get("id") == "server-unknown"
            and unknown_response.get("error", {}).get("code") == -32601
        ):
            tools.append({
                "name": "method_not_found_ok",
                "inputSchema": {"type": "object"}
            })
        write_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools}
        })

if __name__ == "__main__":
    main()
"""


class XcodeMcpClientTests:
    """验证 MCP 客户端基础生命周期和协商约束。"""

    def test_mcp_client_handshake_list_and_call(self) -> None:
        """握手后可发现并调用服务器工具。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            server_file = Path(temp_dir) / "mock_server.py"
            server_file.write_text(MOCK_SERVER_CODE, encoding="utf-8")

            cmd = [sys.executable, "-u", str(server_file)]
            client = McpClient(cmd)

            # Start and handshake
            client.start()
            assert client.protocol_version == "2024-11-05"
            assert client.server_info == {"name": "mock-server", "version": "1.0.0"}
            assert client.instructions == "Use tools carefully."
            assert client.has_server_capability("tools")

            # List tools
            tools = client.list_tools()
            assert len(tools) == 1
            assert tools[0]["name"] == "mock_read_tool"

            # Call tool
            result = client.call_tool("mock_read_tool", {"path": "test.txt"})
            assert "content" in result
            assert result["content"][0]["text"] == "Content of test.txt"

            # Stop client
            client.stop()

    def test_mcp_client_rejects_unsupported_protocol_version(self) -> None:
        """不支持的 server protocolVersion 使握手失败并关闭进程。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            server_file = Path(temp_dir) / "mock_server.py"
            server_file.write_text(
                MOCK_SERVER_CODE.replace("2024-11-05", "2099-01-01"),
                encoding="utf-8",
            )
            client = McpClient([sys.executable, "-u", str(server_file)])

            with pytest.raises(
                RuntimeError,
                match="Unsupported MCP protocol version",
            ):
                client.start()

            assert client.status == "failed"
            assert client.process is None

    def test_mcp_client_sends_latest_supported_protocol_version(self) -> None:
        """initialize 请求发送客户端最新支持的协议版本。"""
        echo_version_code = MOCK_SERVER_CODE.replace(
            '"protocolVersion": "2024-11-05"',
            '"protocolVersion": req.get("params", {}).get("protocolVersion")',
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            server_file = Path(temp_dir) / "mock_server.py"
            server_file.write_text(echo_version_code, encoding="utf-8")
            client = McpClient([sys.executable, "-u", str(server_file)])

            client.start()

            assert client.protocol_version == LATEST_PROTOCOL_VERSION
            client.stop()

    def test_mcp_client_rejects_unnegotiated_tools_feature(self) -> None:
        """服务器未声明 tools capability 时不发送 tools 请求。"""
        client = McpClient(["unused"])
        client._apply_initialize_result(
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "serverInfo": {"name": "no-tools", "version": "1.0.0"},
            }
        )

        with pytest.raises(
            RuntimeError,
            match="did not negotiate 'tools'",
        ):
            client.list_tools()

    def test_mcp_client_collects_paginated_tools(self) -> None:
        """工具发现会按 cursor 聚合所有分页结果。"""
        client = McpClient(["unused"])
        client.server_capabilities = {"tools": {}}
        responses = [
            {
                "tools": [{"name": "first"}],
                "nextCursor": "page-2",
            },
            {
                "tools": [{"name": "second"}],
            },
        ]

        with patch.object(client, "send_request", side_effect=responses) as request:
            tools = client.list_tools(timeout=3.0)

        assert tools == [{"name": "first"}, {"name": "second"}]
        assert request.call_args_list == [
            call("tools/list", {}, timeout=3.0),
            call(
                "tools/list",
                {"cursor": "page-2"},
                timeout=3.0,
            ),
        ]

    def test_mcp_client_rejects_repeated_tool_list_cursor(self) -> None:
        """重复 cursor 会终止分页，避免服务器造成无限循环。"""
        client = McpClient(["unused"])
        client.server_capabilities = {"tools": {}}
        responses = [
            {"tools": [], "nextCursor": "repeated"},
            {"tools": [], "nextCursor": "repeated"},
        ]

        with (
            patch.object(client, "send_request", side_effect=responses),
            pytest.raises(RuntimeError, match="repeated cursor"),
        ):
            client.list_tools()

    @pytest.mark.parametrize(
        "response",
        [
            {"tools": "not-a-list"},
            {"tools": [None]},
            {"tools": [], "nextCursor": ""},
            {"tools": [], "nextCursor": 2},
        ],
    )
    def test_mcp_client_rejects_invalid_tool_list_page(self, response: dict) -> None:
        """分页响应中的 tools 和 nextCursor 必须具有协议要求的类型。"""
        client = McpClient(["unused"])
        client.server_capabilities = {"tools": {}}

        with (
            patch.object(client, "send_request", return_value=response),
            pytest.raises(RuntimeError, match="invalid"),
        ):
            client.list_tools()

    def test_mcp_client_limits_tool_list_pages(self) -> None:
        """分页数量达到保护上限后停止继续请求。"""
        client = McpClient(["unused"])
        client.server_capabilities = {"tools": {}}
        page_index = 0

        def next_page(
            method: str,
            params: dict[str, str],
            timeout: float | None = None,
        ) -> dict[str, object]:
            """返回持续产生新 cursor 的异常分页响应。"""
            nonlocal page_index
            assert method == "tools/list"
            assert timeout is None
            if page_index == 0:
                assert params == {}
            else:
                assert params == {"cursor": f"page-{page_index}"}
            page_index += 1
            return {"tools": [], "nextCursor": f"page-{page_index}"}

        with (
            patch.object(client, "send_request", side_effect=next_page) as request,
            pytest.raises(RuntimeError, match="exceeded"),
        ):
            client.list_tools()

        assert request.call_count == MAX_TOOL_LIST_PAGES

    def test_lazy_client_ref_reconnects_within_attempt_limit(self) -> None:
        """首次连接失败后会在同次调用内有限重连。"""
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

        assert connected is second_client
        assert ref.last_error is None
        assert client_class.call_count == MAX_LAZY_CONNECT_ATTEMPTS
        client_class.assert_called_with(
            ["python", "server.py"],
            {"TOKEN": "value"},
            timeout=4.0,
        )
        first_client.stop.assert_called_once_with()
        second_client.set_tools_changed_callback.assert_called_once_with(callback)
        second_client.start.assert_called_once_with()

    def test_lazy_client_ref_retains_last_error_after_retry_exhaustion(self) -> None:
        """重连耗尽后保留已脱敏的最后一次连接错误。"""
        clients = [MagicMock(spec=McpClient) for _ in range(2)]
        clients[0].start.side_effect = RuntimeError("old failure")
        clients[1].start.side_effect = RuntimeError("Bearer secret-token")
        ref = LazyClientRef(
            "fixture",
            {"command": "server"},
            max_connect_attempts=len(clients),
        )

        with (
            patch(
                "xcode.harness.mcp.client.McpClient",
                side_effect=clients,
            ) as client_class,
            pytest.raises(RuntimeError, match="after 2 attempts"),
        ):
            ref.get_or_create()

        assert client_class.call_count == len(clients)
        assert ref.last_error == "Bearer ****"
        for client in clients:
            client.stop.assert_called_once_with()

    def test_lazy_client_ref_rejects_empty_attempt_budget(self) -> None:
        """连接尝试次数必须至少允许一次启动。"""
        with pytest.raises(ValueError, match="at least 1"):
            LazyClientRef(
                "fixture",
                {"command": "server"},
                max_connect_attempts=0,
            )

    def test_mcp_client_cancels_timed_out_request(self) -> None:
        """请求超时后发送 cancellation notification 并清理活动状态。"""
        client = McpClient(["unused"])
        client.process = MagicMock()
        client.process.poll.return_value = None
        client._running = True

        with (
            patch.object(client, "_write_message") as write_message,
            pytest.raises(TimeoutError, match="tools/call"),
        ):
            client.send_request(
                "tools/call",
                {"name": "slow", "arguments": {}},
                timeout=0.0,
            )

        assert write_message.call_args_list == [
            call(
                {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": 1,
                    "params": {"name": "slow", "arguments": {}},
                }
            ),
            call(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {
                        "requestId": 1,
                        "reason": "Client timeout waiting for tools/call",
                    },
                }
            ),
        ]
        assert client._active_request_ids == set()
        assert client._pending_responses == {}

        client._handle_incoming_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": []},
            }
        )
        assert client._pending_responses == {}

    def test_mcp_client_preserves_timeout_when_cancellation_write_fails(
        self,
    ) -> None:
        """取消通知写入失败不会覆盖原始 timeout 异常。"""
        client = McpClient(["unused"])
        client.process = MagicMock()
        client.process.poll.return_value = None
        client._running = True

        with (
            patch.object(
                client,
                "_write_message",
                side_effect=[None, RuntimeError("closed")],
            ),
            assert_logs(
                "xcode.harness.mcp.client",
                level="WARNING",
            ) as logs,
            pytest.raises(TimeoutError),
        ):
            client.send_request("tools/list", {}, timeout=0.0)

        assert any("request 1 timed out" in message for message in logs.output)

    def test_mcp_client_stop_allows_server_to_exit_after_stdin_eof(self) -> None:
        """服务器在 stdin EOF 后自行退出时不发送终止信号。"""
        client = McpClient(["unused"])
        process = MagicMock()
        client.process = process
        client._running = True
        client._status = "connected"

        client.stop()

        process.stdin.close.assert_called_once()
        process.wait.assert_called_once_with(timeout=SHUTDOWN_GRACE_SECONDS)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()
        assert client.process is None
        assert client.status == "disabled"

    def test_mcp_client_stop_kills_server_after_graceful_and_term_timeout(
        self,
    ) -> None:
        """服务器忽略 EOF 和 TERM 时最终发送 KILL 并等待回收。"""
        client = McpClient(["unused"])
        process = MagicMock()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(["server"], SHUTDOWN_GRACE_SECONDS),
            subprocess.TimeoutExpired(["server"], TERMINATE_GRACE_SECONDS),
            0,
        ]
        client.process = process
        client._running = True

        client.stop()

        process.stdin.close.assert_called_once()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        assert process.wait.call_args_list == [
            call(timeout=SHUTDOWN_GRACE_SECONDS),
            call(timeout=TERMINATE_GRACE_SECONDS),
            call(timeout=KILL_GRACE_SECONDS),
        ]
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()

    def test_mcp_client_dispatches_negotiated_tools_changed_notification(
        self,
    ) -> None:
        """已协商的工具变更通知会在读取线程之外触发刷新。"""
        callback_called = threading.Event()
        client = McpClient(
            ["unused"],
            tools_changed_callback=lambda _client: callback_called.set(),
        )
        client.server_capabilities = {"tools": {"listChanged": True}}
        client._running = True

        client._handle_incoming_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed",
            }
        )

        assert callback_called.wait(timeout=1.0)
        client._running = False

    def test_mcp_client_ignores_undeclared_tools_changed_notification(
        self,
    ) -> None:
        """服务器未声明 listChanged 时忽略工具变更通知。"""
        callback_called = threading.Event()
        client = McpClient(
            ["unused"],
            tools_changed_callback=lambda _client: callback_called.set(),
        )
        client.server_capabilities = {"tools": {}}
        client._running = True

        with assert_logs(
            "xcode.harness.mcp.client",
            level="WARNING",
        ) as logs:
            client._handle_incoming_message(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/tools/list_changed",
                }
            )

        assert not (callback_called.wait(timeout=0.05))
        assert any("without negotiated" in message for message in logs.output)
        client._running = False

    def test_mcp_client_rejects_invalid_server_identity(self) -> None:
        """缺少稳定名称或版本的 serverInfo 无法完成协商。"""
        client = McpClient(["unused"])

        with pytest.raises(RuntimeError, match="invalid serverInfo"):
            client._apply_initialize_result(
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "missing-version"},
                }
            )

    def test_server_requests_interleave_with_normal_response(self) -> None:
        """server request 和 notification 不会被误判为 client response。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            server_file = Path(temp_dir) / "interleaved_server.py"
            server_file.write_text(INTERLEAVED_SERVER_CODE, encoding="utf-8")
            client = McpClient([sys.executable, "-u", str(server_file)])
            client.start()

            with assert_logs(
                "xcode.harness.mcp.client",
                level="WARNING",
            ) as logs:
                tools = client.list_tools()

            client.stop()

        assert [tool["name"] for tool in tools] == ["ping_ok", "method_not_found_ok"]
        assert any("notifications/custom" in message for message in logs.output)


if __name__ == "__main__":
    pytest.main()
