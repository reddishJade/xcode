"""官方 Everything MCP server 的显式外部兼容回归。"""

from __future__ import annotations

import shutil

import pytest

from xcode.harness.mcp.client import McpClient

EVERYTHING_PACKAGE = "@modelcontextprotocol/server-everything@2026.1.26"


@pytest.mark.mcp_external
def test_official_everything_stdio_lifecycle() -> None:
    """覆盖官方 server 的发现、调用、结构化结果、进度和关闭生命周期。"""
    if shutil.which("npx") is None:
        pytest.skip("npx is required for the official MCP server smoke test")

    progress: list[tuple[float, float | None, str | None]] = []
    client = McpClient(
        ["npx", "-y", EVERYTHING_PACKAGE, "stdio"],
        timeout=60.0,
    )
    try:
        client.start()
        tools = client.list_tools(timeout=60.0)
        names = {tool["name"] for tool in tools}
        assert {
            "echo",
            "get-structured-content",
            "trigger-long-running-operation",
        } <= names

        echo = client.call_tool("echo", {"message": "xcode-mcp-smoke"})
        assert "xcode-mcp-smoke" in echo["content"][0]["text"]

        structured = client.call_tool(
            "get-structured-content",
            {"location": "New York"},
        )
        assert isinstance(structured.get("structuredContent"), dict)

        client.call_tool(
            "trigger-long-running-operation",
            {"duration": 0.1, "steps": 2},
            progress_callback=lambda current, total, message: progress.append(
                (current, total, message)
            ),
        )
        assert progress
    finally:
        client.stop()

    assert client.status == "disabled"
