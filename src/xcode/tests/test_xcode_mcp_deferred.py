from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from xcode.harness.mcp.tools import (
    McpServerConfig,
    build_mcp_tools,
    build_mcp_tool_search,
    build_fetch_tools_tool,
)
import pytest


class TestXcodeMcpDeferredLoading:
    """MCP 延迟加载 (defer_loading) 单元测试。"""

    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.local_dir = self.root / ".local"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.local_dir / "mcp_config.json"
        self.cache_path = self.local_dir / "mcp_cache.json"

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_defer_loading_registers_stubs_and_search_tool(self) -> None:
        """测试开启 defer_loading 时注册 Stub 工具和 mcp_tool_search，且不触发进程冷启动。"""
        config = {
            "mcpServers": {
                "heavy_service": {
                    "command": "node",
                    "args": ["heavy.js"],
                    "defer_loading": True,
                }
            }
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        # 写入模拟缓存
        cache = {
            "servers": {
                "heavy_service": {
                    "config_hash": "dummy_hash",
                    "protocol_version": "2025-11-25",
                    "server_info": {"name": "heavy", "version": "1.0.0"},
                    "tools": [
                        {
                            "name": "heavy_calculate",
                            "description": "Perform expensive calculations",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"factor": {"type": "integer"}},
                                "required": ["factor"],
                            },
                        }
                    ],
                }
            }
        }
        self.cache_path.write_text(json.dumps(cache), encoding="utf-8")

        # 由于 compute_config_hash 计算真正的 hash，测试中我们需要 mock 计算或者保持一致
        with patch(
            "xcode.harness.mcp.tools.compute_config_hash", return_value="dummy_hash"
        ):
            tools = build_mcp_tools(self.root)

        # 验证工具是否包含 stub 工具和 search 工具
        tool_names = {t.name for t in tools}
        assert "mcp__heavy_service__heavy_calculate" in tool_names
        assert "mcp_tool_search" in tool_names

        # 验证 stub 的描述和 schema 是缩水/提示版本的
        stub_tool = next(
            t for t in tools if t.name == "mcp__heavy_service__heavy_calculate"
        )
        assert "Parameters unknown until searched" in stub_tool.description
        assert stub_tool.schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    def test_mcp_tool_search_cache_only(self) -> None:
        """测试 mcp_tool_search 仅从缓存中检索，不触发 lazy connection 物理连接。"""
        # 即使没有缓存，搜索工具在搜索时也不会启动 MCP 进程，而是返回 Schema not yet loaded
        search_tool = build_mcp_tool_search(
            self.root, {"heavy_service"}, {"heavy_service": "heavy_service"}
        )

        # 执行搜索
        result = search_tool.handler({"query": "all"})
        assert "Schema not yet loaded" in result
        assert "mcp__heavy_service__fetch_tools" in result

        # 写入缓存后，搜索能成功查到
        cache = {
            "servers": {
                "heavy_service": {
                    "config_hash": "dummy",
                    "protocol_version": "2025-11-25",
                    "server_info": {"name": "heavy", "version": "1.0.0"},
                    "tools": [
                        {
                            "name": "heavy_calculate",
                            "description": "Perform expensive calculations",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "factor": {
                                        "type": "integer",
                                        "description": "Multiplier factor",
                                    }
                                },
                                "required": ["factor"],
                            },
                        }
                    ],
                }
            }
        }
        self.cache_path.write_text(json.dumps(cache), encoding="utf-8")

        result = search_tool.handler({"query": "heavy"})
        assert "mcp__heavy_service__heavy_calculate" in result
        assert "Multiplier factor" in result
        assert "factor (required)" in result

    def test_jit_parameter_validation_in_handler(self) -> None:
        """测试 Stub 执行时 JIT 强校验 required 字段是否正确拦截。"""
        config = {
            "mcpServers": {
                "heavy_service": {
                    "command": "node",
                    "args": ["heavy.js"],
                    "defer_loading": True,
                }
            }
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        cache = {
            "servers": {
                "heavy_service": {
                    "config_hash": "dummy_hash",
                    "protocol_version": "2025-11-25",
                    "server_info": {"name": "heavy", "version": "1.0.0"},
                    "tools": [
                        {
                            "name": "heavy_calculate",
                            "description": "Perform expensive calculations",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"factor": {"type": "integer"}},
                                "required": ["factor"],
                            },
                        }
                    ],
                }
            }
        }
        self.cache_path.write_text(json.dumps(cache), encoding="utf-8")

        with patch(
            "xcode.harness.mcp.tools.compute_config_hash", return_value="dummy_hash"
        ):
            tools = build_mcp_tools(self.root)

        stub_tool = next(
            t for t in tools if t.name == "mcp__heavy_service__heavy_calculate"
        )

        # 缺省 factor 参数调用，应被 JIT 校验直接拦截
        with pytest.raises(ValueError) as exc_info:
            stub_tool.handler({"args": {}})
        assert "Missing required parameters" in str(exc_info.value)

        # 提供正确参数调用时，应透传给后台 lazy connection 执行（mock 掉 mcp client）
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "content": [{"type": "text", "text": "result: 42"}]
        }

        with patch(
            "xcode.harness.mcp.client.LazyClientRef.get_or_create",
            return_value=mock_client,
        ):
            output = stub_tool.handler({"factor": 5})
            assert output == "result: 42"
            mock_client.call_tool.assert_called_with(
                "heavy_calculate", {"factor": 5}, timeout=None
            )

    def test_fetch_tools_bootstrap_tool(self) -> None:
        """测试 fetch_tools 引导工具是否能正常冷启动拉取列表并缓存。"""
        validated = McpServerConfig(
            name="heavy_service",
            command=("node",),
            args=("heavy.js",),
        )
        bootstrap_tool = build_fetch_tools_tool(self.root, "heavy_service", validated)

        # 模拟 McpClient 启动与拉取
        mock_client = MagicMock()
        mock_client.protocol_version = "2025-11-25"
        mock_client.server_info = {"name": "heavy", "version": "1.0.0"}
        mock_client.list_tools.return_value = [
            {
                "name": "calc",
                "description": "quick calculation",
                "inputSchema": {"type": "object"},
            }
        ]

        with patch("xcode.harness.mcp.client.McpClient", return_value=mock_client):
            result = bootstrap_tool.handler({})
            assert "Successfully fetched 1 tools" in result
            assert self.cache_path.exists()

            # 验证缓存内容
            cache_data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            entry = cache_data["servers"]["heavy_service"]
            assert len(entry["tools"]) == 1
            assert entry["tools"][0]["name"] == "calc"
            assert entry["protocol_version"] == "2025-11-25"
            assert entry["server_info"] == {"name": "heavy", "version": "1.0.0"}


if __name__ == "__main__":
    pytest.main()
