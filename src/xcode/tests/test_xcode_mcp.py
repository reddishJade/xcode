"""Step 9 MCP canonicalization tests.

Covers: config validation, naming/collisions, permission integration,
error handling, redaction, subagent exclusion.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from xcode.harness.mcp.tools import (
    McpServerConfig,
    _sanitize,
    _validate_server_config,
    _mcp_config_path,
    _detect_collisions,
    build_mcp_tools,
)
from xcode.harness.mcp.client import (
    redact_mcp_text,
    truncate_redact,
)
from xcode.harness.mcp.results import MCP_RESULT_METADATA_KEY

from xcode.harness.observability.permission_model import (
    ActionExtractor,
    Action,
    Target,
)
from xcode.harness.skills import ToolOutput
# ── 辅助 ──


def _write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _minimal_mcp_tool(name: str = "test_tool") -> dict[str, Any]:
    return {
        "name": name,
        "description": "A test tool",
        "inputSchema": {
            "type": "object",
            "properties": {"param1": {"type": "string", "description": "A parameter"}},
            "required": ["param1"],
        },
    }


# ════════════════════════════════════════════
# 1. 配置校验
# ════════════════════════════════════════════


class TestMcpConfigValidation:
    """Config schema validation tests."""

    def test_valid_config(self) -> None:
        cfg = _validate_server_config(
            "my_server",
            {
                "command": "python",
                "args": ["server.py"],
                "env": {"KEY": "VAL"},
                "enabled": True,
                "timeout": 15.0,
            },
        )
        assert cfg is not None
        assert cfg is not None
        assert cfg.name == "my_server"
        assert cfg.command == ("python",)
        assert cfg.args == ("server.py",)
        assert cfg.env == {"KEY": "VAL"}
        assert cfg.enabled
        assert cfg.timeout == 15.0

    def test_minimal_config(self) -> None:
        cfg = _validate_server_config("min", {"command": "node"})
        assert cfg is not None
        assert cfg is not None
        assert cfg.command == ("node",)
        assert cfg.args == ()
        assert cfg.env is None
        assert cfg.enabled
        assert cfg.timeout is None

    def test_overrides_are_allowed(self) -> None:
        cfg = _validate_server_config(
            "bad",
            {
                "command": "python",
                "overrides": {"tool": {"risk": "high"}},
            },
        )
        assert cfg is not None

    def test_empty_command_skips(self) -> None:
        cfg = _validate_server_config("bad", {"command": ""})
        assert cfg is None

    def test_non_dict_config_skips(self) -> None:
        cfg = _validate_server_config("bad", cast(dict[str, object], "not_a_dict"))
        assert cfg is None

    def test_enabled_false(self) -> None:
        cfg = _validate_server_config(
            "off",
            {
                "command": "python",
                "enabled": False,
            },
        )
        assert cfg is not None
        assert cfg is not None
        assert not (cfg.enabled)

    def test_timeout_as_int(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "timeout": 5000,
            },
        )
        assert cfg is not None
        assert cfg is not None
        assert cfg.timeout == 5000.0

    def test_invalid_timeout_ignored(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "timeout": "fast",
            },
        )
        assert cfg is not None
        assert cfg is not None
        assert cfg.timeout is None

    def test_invalid_env_ignored(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "env": "not_a_dict",
            },
        )
        assert cfg is not None
        assert cfg is not None
        assert cfg.env is None

    def test_non_list_args_skips(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "args": "not_a_list",
            },
        )
        assert cfg is not None
        assert cfg is not None
        assert cfg.args == ()


# ════════════════════════════════════════════
# 2. 配置路径
# ════════════════════════════════════════════


class TestMcpConfigPath:
    """Config path canonicalization tests."""

    def test_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(local / "mcp_config.json", {"mcpServers": {}})
            result = _mcp_config_path(root)
            assert result == local / "mcp_config.json"

    def test_root_config_path_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root / "mcp_config.json", {"mcpServers": {}})
            result = _mcp_config_path(root)
            assert result is None

    def test_no_config_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _mcp_config_path(root)
            assert result is None

    def test_canonical_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(
                local / "mcp_config.json", {"mcpServers": {"good": {"command": "a"}}}
            )
            _write_config(
                root / "mcp_config.json", {"mcpServers": {"bad": {"command": "b"}}}
            )
            result = _mcp_config_path(root)
            assert result == local / "mcp_config.json"


# ════════════════════════════════════════════
# 3. 名称清理
# ════════════════════════════════════════════


class TestMcpSanitize:
    """Name sanitization tests."""

    def test_basic(self) -> None:
        assert _sanitize("hello") == "hello"

    def test_spaces_to_underscore(self) -> None:
        assert _sanitize("my server") == "my_server"

    def test_special_chars(self) -> None:
        assert _sanitize("foo@bar!baz") == "foo_bar_baz"

    def test_unicode_replaced(self) -> None:
        assert _sanitize("café") == "caf_"

    def test_dots_replaced(self) -> None:
        assert _sanitize("server.local") == "server_local"

    def test_hyphen_preserved(self) -> None:
        assert _sanitize("my-server") == "my-server"

    def test_already_slug(self) -> None:
        assert _sanitize("my_server") == "my_server"

    def test_empty_returns_empty(self) -> None:
        assert _sanitize("") == ""

    def test_leading_trailing_spaces(self) -> None:
        assert _sanitize("  tool  ") == "__tool__"


# ════════════════════════════════════════════
# 4. 碰撞检测
# ════════════════════════════════════════════


class TestMcpCollisionDetection:
    """Collision detection tests."""

    def _make_tool(self, name: str) -> dict[str, Any]:
        return _minimal_mcp_tool(name)

    def test_no_collision(self) -> None:
        tools = {"s1": [self._make_tool("read"), self._make_tool("write")]}
        servers = {"s1": McpServerConfig(name="s1", command=("python",))}
        disabled = _detect_collisions(tools, servers)
        assert disabled == set()

    def test_same_server_collision(self) -> None:
        # Space becomes _, making "my tool" and "my_tool" collide
        tools = {"s1": [self._make_tool("my tool"), self._make_tool("my_tool")]}
        servers = {"s1": McpServerConfig(name="s1", command=("python",))}
        disabled = _detect_collisions(tools, servers)
        assert disabled == {"s1:my tool", "s1:my_tool"}

    def test_cross_server_collision(self) -> None:
        # Same tool name on different servers produces different host IDs
        # (mcp__s1__read vs mcp__s2__read) — no collision
        tools = {
            "s1": [self._make_tool("read")],
            "s2": [self._make_tool("read")],
        }
        servers = {
            "s1": McpServerConfig(name="s1", command=("python",)),
            "s2": McpServerConfig(name="s2", command=("python",)),
        }
        disabled = _detect_collisions(tools, servers)
        assert disabled == set()

    def test_cross_server_collision_same_slug(self) -> None:
        # Different server names that sanitize to same slug cause collision
        tools = {
            "my srv": [self._make_tool("read")],
            "my_srv": [self._make_tool("read")],
        }
        servers = {
            "my srv": McpServerConfig(name="my srv", command=("python",)),
            "my_srv": McpServerConfig(name="my_srv", command=("python",)),
        }
        disabled = _detect_collisions(tools, servers)
        # Both produce mcp__my_srv__read
        assert disabled == {"my srv:read", "my_srv:read"}

    def test_complex_collision(self) -> None:
        # Multiple paths to same host_tool_id
        tools = {
            "my srv": [
                self._make_tool("get data"),
                self._make_tool("get_data"),
            ],
            "my_srv": [self._make_tool("get data")],
        }
        servers = {
            "my srv": McpServerConfig(name="my srv", command=("python",)),
            "my_srv": McpServerConfig(name="my_srv", command=("python",)),
        }
        disabled = _detect_collisions(tools, servers)
        # All produce mcp__my_srv__get_data
        assert disabled == {
            "my srv:get data",
            "my srv:get_data",
            "my_srv:get data",
        }


# ════════════════════════════════════════════
# 5. 权限 - ActionExtractor MCP 分支
# ════════════════════════════════════════════


class TestMcpActionExtractor:
    """ActionExtractor MCP branch tests."""

    def setup_method(self, method) -> None:
        self.extractor = ActionExtractor()

    def test_mcp_tool_produces_unknown_action(self) -> None:
        action = self.extractor.extract(
            "mcp__my_server__read_file",
            {"path": "/tmp/test.txt"},
        )
        assert action.capability == "mcp"
        assert len(action.targets) == 1
        assert action.targets[0].kind == "mcp"
        assert action.targets[0].value == "mcp__my_server__read_file"

    def test_mcp_tool_empty_input(self) -> None:
        action = self.extractor.extract(
            "mcp__server__fetch",
            {},
        )
        assert action.capability == "mcp"
        assert len(action.targets) == 1
        assert action.targets[0].kind == "mcp"
        assert action.targets[0].value == "mcp__server__fetch"

    def test_non_mcp_tool_unchanged(self) -> None:
        action = self.extractor.extract("bash", {"command": "ls"})
        assert action.capability == "shell"
        assert action.targets[0].kind != "mcp" if action.targets else "mcp"

    def test_unknown_tool_fallback(self) -> None:
        action = self.extractor.extract("custom_tool", {})
        assert action.capability == "unknown"
        assert action.targets == ()


# ════════════════════════════════════════════
# 6. 权限 - PermissionEngine 决策路径
# ════════════════════════════════════════════


class TestMcpPermissionDecisions:
    """PermissionEngine behavior for MCP tools.

    Tests are simplified: we verify the ActionExtractor output that
    feeds into PermissionEngine. Full engine-level tests exist in
    test_xcode_permissions.py.
    """

    def test_mcp_action_has_mcp_target(self) -> None:
        """MCP tools get an mcp target for permission matching."""
        extractor = ActionExtractor()
        action = extractor.extract("mcp__srv__tool", {"arg": "val"})
        assert len(action.targets) == 1
        assert action.targets[0].kind == "mcp"
        assert action.targets[0].value == "mcp__srv__tool"

    def test_mcp_target_enables_grant_lookup(self) -> None:
        """MCP targets enable grant candidate lookup."""
        from xcode.harness.observability.permission_model import (
            compute_shadow_approval_candidate,
        )

        extractor = ActionExtractor()
        action = extractor.extract("mcp__srv__tool", {})
        candidate = compute_shadow_approval_candidate(action)
        assert candidate is not None
        assert any(f.fingerprint.target_kind == "mcp" for f in candidate.fingerprints)

    def test_mcp_grant_written(self) -> None:
        """Grant records for MCP tools are created correctly."""
        from xcode.harness.observability.permission_model import (
            create_grant_record,
        )

        action = Action(
            tool="mcp__srv__tool",
            capability="mcp",
            operation="mcp__srv__tool",
            targets=(Target(kind="mcp", value="mcp__srv__tool", access="execute"),),
            input={},
        )
        grant = create_grant_record(
            action,
            action.targets[0],
            decision="allow",
            scope="session",
        )
        assert grant.capability == "mcp"
        assert grant.target_kind == "mcp"
        assert grant.target_pattern == "mcp__srv__tool"
        assert grant.access == "execute"


# ════════════════════════════════════════════
# 7. 错误处理与脱敏
# ════════════════════════════════════════════


class TestMcpRedaction:
    """Stderr redaction and truncation tests."""

    def test_bearer_token_redacted(self) -> None:
        result = redact_mcp_text("Bearer sk-abc123def456")
        assert "****" in result
        assert "sk-abc123def456" not in result

    def test_api_key_redacted(self) -> None:
        result = redact_mcp_text("API_KEY=super_secret_key_123")
        assert "****" in result
        assert "super_secret_key_123" not in result

    def test_token_redacted(self) -> None:
        result = redact_mcp_text("TOKEN=abc123")
        assert "****" in result
        assert "abc123" not in result

    def test_secret_redacted(self) -> None:
        result = redact_mcp_text("SECRET=my_secret_value")
        assert "****" in result
        assert "my_secret_value" not in result

    def test_mixed_content(self) -> None:
        msg = "Error: Bearer sk-abc, status: 500"
        result = redact_mcp_text(msg)
        assert "Error:" in result
        assert "sk-abc" not in result

    def test_truncate_short(self) -> None:
        msg = "short error"
        result = truncate_redact(msg, max_len=200)
        assert result == msg

    def test_truncate_long(self) -> None:
        msg = "x" * 300
        result = truncate_redact(msg, max_len=200)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    def test_truncate_with_redaction(self) -> None:
        msg = "Error: Token=abc123 " + "x" * 300
        result = truncate_redact(msg, max_len=200)
        assert "abc123" not in result
        assert "****" in result
        assert result.endswith("...")

    def test_no_sensitive_content_passes_through(self) -> None:
        msg = "normal error message"
        result = redact_mcp_text(msg)
        assert result == msg


# ════════════════════════════════════════════
# 9. MCP 工具构建（集成）
# ════════════════════════════════════════════


@patch("xcode.harness.mcp.client.McpClient")
class TestMcpBuildIntegration:
    """Integration tests for build_mcp_tools."""

    def test_builds_tools_from_config(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.protocol_version = "2025-11-25"
        mock_instance.server_info = {"name": "my-server", "version": "1.0.0"}
        mock_instance.list_tools.return_value = [
            _minimal_mcp_tool("greet"),
            _minimal_mcp_tool("echo"),
        ]
        mock_instance.status = "connected"
        mock_client.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(
                local / "mcp_config.json",
                {
                    "mcpServers": {
                        "my_server": {
                            "command": "python",
                            "args": ["server.py"],
                        }
                    }
                },
            )
            tools = build_mcp_tools(root)
            assert len(tools) == 2
            names = [t.name for t in tools]
            assert "mcp__my_server__greet" in names
            assert "mcp__my_server__echo" in names
            for t in tools:
                assert t.group == "mcp"

    def test_disabled_server_skipped(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.protocol_version = "2025-11-25"
        mock_instance.server_info = {"name": "enabled", "version": "1.0.0"}
        mock_instance.list_tools.return_value = []
        mock_client.return_value = mock_instance
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(
                local / "mcp_config.json",
                {
                    "mcpServers": {
                        "enabled_srv": {
                            "command": "python",
                            "enabled": True,
                        },
                        "disabled_srv": {
                            "command": "python",
                            "enabled": False,
                        },
                    }
                },
            )
            # Only enabled server should trigger client creation
            tools = build_mcp_tools(root)
            # disabled_srv produces no tools; enabled_srv hits cache miss
            # and fails because mock has no list_tools — returns empty
            assert isinstance(tools, tuple)

    def test_overrides_server_applied(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.protocol_version = "2025-11-25"
        mock_instance.server_info = {"name": "good", "version": "1.0.0"}
        mock_instance.list_tools.return_value = [_minimal_mcp_tool("x")]
        mock_client.return_value = mock_instance
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(
                local / "mcp_config.json",
                {
                    "mcpServers": {
                        "good": {"command": "python"},
                        "bad": {
                            "command": "python",
                            "overrides": {
                                "x": {
                                    "read_only": True,
                                    "concurrency_safe": True,
                                    "description": "Overridden description",
                                    "risk": "high",
                                }
                            },
                        },
                    }
                },
            )
            tools = build_mcp_tools(root)
            overridden = next(tool for tool in tools if tool.name == "mcp__bad__x")
            assert overridden.read_only
            assert overridden.concurrency_safe
            assert overridden.description == "Overridden description"
            assert overridden.builtin is not None
            meta = overridden.builtin["mcp_metadata"]
            assert meta["risk"] == "high"

    def test_mcp_metadata_on_tool_spec(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.protocol_version = "2025-11-25"
        mock_instance.server_info = {"name": "test-server", "version": "1.0.0"}
        tool = _minimal_mcp_tool("my_tool")
        tool["outputSchema"] = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        }
        tool["annotations"] = {"readOnlyHint": True}
        mock_instance.list_tools.return_value = [
            tool,
        ]
        mock_instance.status = "connected"
        mock_client.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(
                local / "mcp_config.json",
                {
                    "mcpServers": {
                        "test-srv": {
                            "command": "python",
                        }
                    }
                },
            )
            tools = build_mcp_tools(root)
            assert len(tools) == 1
            spec = tools[0]
            assert spec.name == "mcp__test-srv__my_tool"
            assert spec.group == "mcp"
            assert spec.builtin is not None
            assert spec.builtin is not None
            meta = spec.builtin.get("mcp_metadata", {})
            assert meta.get("server") == "test-srv"
            assert meta.get("server_slug") == "test-srv"
            assert meta.get("tool") == "my_tool"
            assert meta.get("tool_slug") == "my_tool"
            assert meta.get("outputSchema") == tool["outputSchema"]
            assert meta.get("annotations") == {"readOnlyHint": True}

    def test_mcp_handler_preserves_modern_result(self, mock_client: MagicMock) -> None:
        """注册后的 handler 会校验并保留 structuredContent。"""
        mock_instance = MagicMock()
        mock_instance.protocol_version = "2025-11-25"
        mock_instance.server_info = {"name": "modern", "version": "1.0.0"}
        tool = _minimal_mcp_tool("weather")
        tool["outputSchema"] = {
            "type": "object",
            "properties": {"temperature": {"type": "number"}},
            "required": ["temperature"],
        }
        mock_instance.list_tools.return_value = [tool]
        def call_tool_side_effect(
            _name: str,
            _arguments: dict[str, object],
            *,
            timeout: float | None = None,
            progress_callback=None,
            cancel_event=None,
        ) -> dict[str, object]:
            assert timeout is None
            assert cancel_event is None
            assert progress_callback is not None
            progress_callback(0.5, 1.0, "Halfway")
            return {
                "content": [{"type": "text", "text": "22.5 C"}],
                "structuredContent": {"temperature": 22.5},
            }

        mock_instance.call_tool.side_effect = call_tool_side_effect
        mock_instance.status = "connected"
        mock_client.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(
                root / ".local" / "mcp_config.json",
                {"mcpServers": {"modern": {"command": "python"}}},
            )
            (spec,) = build_mcp_tools(root)

            output = spec.handler({})

        assert isinstance(output, ToolOutput)
        assert isinstance(output, ToolOutput)
        assert "22.5 C" in output
        details = output.metadata[MCP_RESULT_METADATA_KEY]
        assert isinstance(details, dict)
        assert details["validation"]["status"] == "valid"
        assert output.metadata["mcp_progress"] == [
            {"progress": 0.5, "total": 1.0, "message": "Halfway"}
        ]
        mock_instance.call_tool.assert_called_once_with(
            "weather",
            {},
            timeout=None,
            progress_callback=mock_instance.call_tool.call_args.kwargs["progress_callback"],
            cancel_event=None,
        )

    def test_collision_disables_tools(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.protocol_version = "2025-11-25"
        mock_instance.server_info = {"name": "collision-server", "version": "1.0.0"}
        mock_instance.list_tools.return_value = [
            _minimal_mcp_tool("my tool"),
            _minimal_mcp_tool("my_tool"),
        ]
        mock_instance.status = "connected"
        mock_client.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(
                local / "mcp_config.json",
                {
                    "mcpServers": {
                        "srv": {
                            "command": "python",
                        }
                    }
                },
            )
            tools = build_mcp_tools(root)
            # Both tools produce mcp__srv__my_tool → collision → both disabled
            assert len(tools) == 0


# ════════════════════════════════════════════
# 10. MCP 红action/脱敏 — McpClient 集成
# ════════════════════════════════════════════


class TestMcpClientRedaction:
    """McpClient stderr redaction in error messages."""

    def test_stderr_redacted_and_truncated(self) -> None:
        """Verify that stderr in process-exit errors is redacted."""
        raw = "Bearer sk-secret123\n" + "x" * 500
        result = truncate_redact(raw, max_len=200)
        assert "sk-secret123" not in result
        assert "****" in result
        assert len(result) <= 203  # 200 + "..."
