"""MCP 客户端握手、协商和工具调用测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from xcode.harness.mcp.client import (
    LATEST_PROTOCOL_VERSION,
    MAX_TOOL_LIST_PAGES,
    McpClient,
)

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


class XcodeMcpClientTests(unittest.TestCase):
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
            self.assertEqual(client.protocol_version, "2024-11-05")
            self.assertEqual(
                client.server_info,
                {"name": "mock-server", "version": "1.0.0"},
            )
            self.assertEqual(client.instructions, "Use tools carefully.")
            self.assertTrue(client.has_server_capability("tools"))

            # List tools
            tools = client.list_tools()
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0]["name"], "mock_read_tool")

            # Call tool
            result = client.call_tool("mock_read_tool", {"path": "test.txt"})
            self.assertIn("content", result)
            self.assertEqual(result["content"][0]["text"], "Content of test.txt")

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

            with self.assertRaisesRegex(
                RuntimeError,
                "Unsupported MCP protocol version",
            ):
                client.start()

            self.assertEqual(client.status, "failed")
            self.assertIsNone(client.process)

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

            self.assertEqual(client.protocol_version, LATEST_PROTOCOL_VERSION)
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

        with self.assertRaisesRegex(
            RuntimeError,
            "did not negotiate 'tools'",
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

        self.assertEqual(tools, [{"name": "first"}, {"name": "second"}])
        self.assertEqual(
            request.call_args_list,
            [
                call("tools/list", {}, timeout=3.0),
                call(
                    "tools/list",
                    {"cursor": "page-2"},
                    timeout=3.0,
                ),
            ],
        )

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
            self.assertRaisesRegex(RuntimeError, "repeated cursor"),
        ):
            client.list_tools()

    def test_mcp_client_rejects_invalid_tool_list_page(self) -> None:
        """分页响应中的 tools 和 nextCursor 必须具有协议要求的类型。"""
        client = McpClient(["unused"])
        client.server_capabilities = {"tools": {}}

        invalid_responses = (
            {"tools": "not-a-list"},
            {"tools": [None]},
            {"tools": [], "nextCursor": ""},
            {"tools": [], "nextCursor": 2},
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                with (
                    patch.object(client, "send_request", return_value=response),
                    self.assertRaisesRegex(RuntimeError, "invalid"),
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
            self.assertEqual(method, "tools/list")
            self.assertIsNone(timeout)
            if page_index == 0:
                self.assertEqual(params, {})
            else:
                self.assertEqual(
                    params,
                    {"cursor": f"page-{page_index}"},
                )
            page_index += 1
            return {"tools": [], "nextCursor": f"page-{page_index}"}

        with (
            patch.object(client, "send_request", side_effect=next_page) as request,
            self.assertRaisesRegex(RuntimeError, "exceeded"),
        ):
            client.list_tools()

        self.assertEqual(request.call_count, MAX_TOOL_LIST_PAGES)

    def test_mcp_client_rejects_invalid_server_identity(self) -> None:
        """缺少稳定名称或版本的 serverInfo 无法完成协商。"""
        client = McpClient(["unused"])

        with self.assertRaisesRegex(RuntimeError, "invalid serverInfo"):
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

            with self.assertLogs(
                "xcode.harness.mcp.client",
                level="WARNING",
            ) as logs:
                tools = client.list_tools()

            client.stop()

        self.assertEqual(
            [tool["name"] for tool in tools],
            ["ping_ok", "method_not_found_ok"],
        )
        self.assertTrue(
            any("notifications/custom" in message for message in logs.output)
        )


if __name__ == "__main__":
    unittest.main()
