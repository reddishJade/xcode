"""MCP stdio 客户端与工具集成。"""

from .client import LazyClientRef, McpClient
from .tools import McpServerConfig, McpToolMetadata, build_mcp_tools

__all__ = [
    "LazyClientRef",
    "McpClient",
    "McpServerConfig",
    "McpToolMetadata",
    "build_mcp_tools",
]
