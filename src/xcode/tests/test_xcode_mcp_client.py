from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
from xcode.experimental.mcp_client import McpClient

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
                            "capabilities": {},
                            "serverInfo": {"name": "mock-server", "version": "1.0.0"}
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
    def test_mcp_client_handshake_list_and_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_file = Path(temp_dir) / "mock_server.py"
            server_file.write_text(MOCK_SERVER_CODE, encoding="utf-8")

            cmd = [sys.executable, "-u", str(server_file)]
            client = McpClient(cmd)

            # Start and handshake
            client.start()

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


if __name__ == "__main__":
    unittest.main()
