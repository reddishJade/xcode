"""MCP 客户端握手、协商和工具调用测试。"""

from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
from xcode.harness.mcp.client import LATEST_PROTOCOL_VERSION, McpClient

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


if __name__ == "__main__":
    unittest.main()
