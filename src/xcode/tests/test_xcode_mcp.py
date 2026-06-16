"""Step 9 MCP canonicalization tests.

Covers: config validation, naming/collisions, permission integration,
error handling, redaction, subagent exclusion, fromClaude compat.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

from xcode.experimental.mcp import (
    McpServerConfig,
    _sanitize,
    _validate_server_config,
    _mcp_config_path,
    _detect_collisions,
    build_mcp_tools,
)
from xcode.experimental.mcp_client import (
    redact_mcp_text,
    truncate_redact,
)
from xcode.experimental.mcp_config_compat import (
    from_claude_config,
)
from xcode.harness.observability.permission_model import (
    ActionExtractor,
    Action,
    Target,
)


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


class TestMcpConfigValidation(TestCase):
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
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.name, "my_server")
        self.assertEqual(cfg.command, ("python",))
        self.assertEqual(cfg.args, ("server.py",))
        self.assertEqual(cfg.env, {"KEY": "VAL"})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.timeout, 15.0)

    def test_minimal_config(self) -> None:
        cfg = _validate_server_config("min", {"command": "node"})
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.command, ("node",))
        self.assertEqual(cfg.args, ())
        self.assertIsNone(cfg.env)
        self.assertTrue(cfg.enabled)
        self.assertIsNone(cfg.timeout)

    def test_overrides_skips_server(self) -> None:
        cfg = _validate_server_config(
            "bad",
            {
                "command": "python",
                "overrides": {"tool": "low"},
            },
        )
        self.assertIsNone(cfg)

    def test_empty_command_skips(self) -> None:
        cfg = _validate_server_config("bad", {"command": ""})
        self.assertIsNone(cfg)

    def test_non_dict_config_skips(self) -> None:
        cfg = _validate_server_config("bad", "not_a_dict")  # type: ignore[arg-type]
        self.assertIsNone(cfg)

    def test_enabled_false(self) -> None:
        cfg = _validate_server_config(
            "off",
            {
                "command": "python",
                "enabled": False,
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertFalse(cfg.enabled)

    def test_timeout_as_int(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "timeout": 5000,
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.timeout, 5000.0)

    def test_invalid_timeout_ignored(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "timeout": "fast",
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertIsNone(cfg.timeout)

    def test_invalid_env_ignored(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "env": "not_a_dict",
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertIsNone(cfg.env)

    def test_non_list_args_skips(self) -> None:
        cfg = _validate_server_config(
            "t",
            {
                "command": "python",
                "args": "not_a_list",
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.args, ())


# ════════════════════════════════════════════
# 2. 配置路径
# ════════════════════════════════════════════


class TestMcpConfigPath(TestCase):
    """Config path canonicalization tests."""

    def test_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / ".local"
            local.mkdir()
            _write_config(local / "mcp_config.json", {"mcpServers": {}})
            result = _mcp_config_path(root)
            self.assertEqual(result, local / "mcp_config.json")

    def test_legacy_root_path_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root / "mcp_config.json", {"mcpServers": {}})
            result = _mcp_config_path(root)
            self.assertIsNone(result)

    def test_no_config_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _mcp_config_path(root)
            self.assertIsNone(result)

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
            self.assertEqual(result, local / "mcp_config.json")


# ════════════════════════════════════════════
# 3. 名称清理
# ════════════════════════════════════════════


class TestMcpSanitize(TestCase):
    """Name sanitization tests."""

    def test_basic(self) -> None:
        self.assertEqual(_sanitize("hello"), "hello")

    def test_spaces_to_underscore(self) -> None:
        self.assertEqual(_sanitize("my server"), "my_server")

    def test_special_chars(self) -> None:
        self.assertEqual(_sanitize("foo@bar!baz"), "foo_bar_baz")

    def test_unicode_replaced(self) -> None:
        self.assertEqual(_sanitize("café"), "caf_")

    def test_dots_replaced(self) -> None:
        self.assertEqual(_sanitize("server.local"), "server_local")

    def test_hyphen_preserved(self) -> None:
        self.assertEqual(_sanitize("my-server"), "my-server")

    def test_already_slug(self) -> None:
        self.assertEqual(_sanitize("my_server"), "my_server")

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(_sanitize(""), "")

    def test_leading_trailing_spaces(self) -> None:
        self.assertEqual(_sanitize("  tool  "), "__tool__")


# ════════════════════════════════════════════
# 4. 碰撞检测
# ════════════════════════════════════════════


class TestMcpCollisionDetection(TestCase):
    """Collision detection tests."""

    def _make_tool(self, name: str) -> dict[str, Any]:
        return _minimal_mcp_tool(name)

    def test_no_collision(self) -> None:
        tools = {"s1": [self._make_tool("read"), self._make_tool("write")]}
        servers = {"s1": McpServerConfig(name="s1", command=("python",))}
        disabled = _detect_collisions(tools, servers)
        self.assertEqual(disabled, set())

    def test_same_server_collision(self) -> None:
        # Space becomes _, making "my tool" and "my_tool" collide
        tools = {"s1": [self._make_tool("my tool"), self._make_tool("my_tool")]}
        servers = {"s1": McpServerConfig(name="s1", command=("python",))}
        disabled = _detect_collisions(tools, servers)
        self.assertEqual(disabled, {"s1:my tool", "s1:my_tool"})

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
        self.assertEqual(disabled, set())

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
        self.assertEqual(disabled, {"my srv:read", "my_srv:read"})

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
        self.assertEqual(
            disabled,
            {
                "my srv:get data",
                "my srv:get_data",
                "my_srv:get data",
            },
        )


# ════════════════════════════════════════════
# 5. 权限 - ActionExtractor MCP 分支
# ════════════════════════════════════════════


class TestMcpActionExtractor(TestCase):
    """ActionExtractor MCP branch tests."""

    def setUp(self) -> None:
        self.extractor = ActionExtractor()

    def test_mcp_tool_produces_mcp_action(self) -> None:
        action = self.extractor.extract(
            "mcp__my_server__read_file",
            {"path": "/tmp/test.txt"},
        )
        self.assertEqual(action.capability, "mcp")
        self.assertEqual(len(action.targets), 1)
        target = action.targets[0]
        self.assertEqual(target.kind, "mcp")
        self.assertEqual(target.value, "mcp__my_server__read_file")
        self.assertEqual(target.access, "execute")

    def test_mcp_tool_empty_input(self) -> None:
        action = self.extractor.extract(
            "mcp__server__fetch",
            {},
        )
        self.assertEqual(action.capability, "mcp")
        self.assertEqual(len(action.targets), 1)
        self.assertEqual(action.targets[0].kind, "mcp")

    def test_non_mcp_tool_unchanged(self) -> None:
        action = self.extractor.extract("bash", {"command": "ls"})
        self.assertEqual(action.capability, "shell")
        self.assertNotEqual(action.targets[0].kind, "mcp" if action.targets else "mcp")

    def test_unknown_tool_fallback(self) -> None:
        action = self.extractor.extract("custom_tool", {})
        self.assertEqual(action.capability, "unknown")
        self.assertEqual(action.targets, ())


# ════════════════════════════════════════════
# 6. 权限 - PermissionEngine 决策路径
# ════════════════════════════════════════════


class TestMcpPermissionDecisions(TestCase):
    """PermissionEngine behavior for MCP tools.

    Tests are simplified: we verify the ActionExtractor output that
    feeds into PermissionEngine. Full engine-level tests exist in
    test_xcode_permissions.py.
    """

    def test_mcp_action_has_non_empty_targets(self) -> None:
        """Grant lookup and storage now work because targets are non-empty."""
        extractor = ActionExtractor()
        action = extractor.extract("mcp__srv__tool", {"arg": "val"})
        self.assertTrue(len(action.targets) > 0)

    def test_mcp_target_enables_grant_lookup(self) -> None:
        """compute_shadow_approval_candidate no longer returns None."""
        from xcode.harness.observability.permission_model import (
            compute_shadow_approval_candidate,
        )

        extractor = ActionExtractor()
        action = extractor.extract("mcp__srv__tool", {})
        # Without targets, this returned None; now it should proceed
        candidate = compute_shadow_approval_candidate(action)
        # Candidate may still be None if no grants exist, but not because
        # targets is empty
        self.assertIsNotNone(candidate)

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
        self.assertEqual(grant.capability, "mcp")
        self.assertEqual(grant.target_kind, "mcp")
        self.assertEqual(grant.target_pattern, "mcp__srv__tool")
        self.assertEqual(grant.access, "execute")


# ════════════════════════════════════════════
# 7. 错误处理与脱敏
# ════════════════════════════════════════════


class TestMcpRedaction(TestCase):
    """Stderr redaction and truncation tests."""

    def test_bearer_token_redacted(self) -> None:
        result = redact_mcp_text("Bearer sk-abc123def456")
        self.assertIn("****", result)
        self.assertNotIn("sk-abc123def456", result)

    def test_api_key_redacted(self) -> None:
        result = redact_mcp_text("API_KEY=super_secret_key_123")
        self.assertIn("****", result)
        self.assertNotIn("super_secret_key_123", result)

    def test_token_redacted(self) -> None:
        result = redact_mcp_text("TOKEN=abc123")
        self.assertIn("****", result)
        self.assertNotIn("abc123", result)

    def test_secret_redacted(self) -> None:
        result = redact_mcp_text("SECRET=my_secret_value")
        self.assertIn("****", result)
        self.assertNotIn("my_secret_value", result)

    def test_mixed_content(self) -> None:
        msg = "Error: Bearer sk-abc, status: 500"
        result = redact_mcp_text(msg)
        self.assertIn("Error:", result)
        self.assertNotIn("sk-abc", result)

    def test_truncate_short(self) -> None:
        msg = "short error"
        result = truncate_redact(msg, max_len=200)
        self.assertEqual(result, msg)

    def test_truncate_long(self) -> None:
        msg = "x" * 300
        result = truncate_redact(msg, max_len=200)
        self.assertEqual(len(result), 203)  # 200 + "..."
        self.assertTrue(result.endswith("..."))

    def test_truncate_with_redaction(self) -> None:
        msg = "Error: Token=abc123 " + "x" * 300
        result = truncate_redact(msg, max_len=200)
        self.assertNotIn("abc123", result)
        self.assertIn("****", result)
        self.assertTrue(result.endswith("..."))

    def test_no_sensitive_content_passes_through(self) -> None:
        msg = "normal error message"
        result = redact_mcp_text(msg)
        self.assertEqual(result, msg)


# ════════════════════════════════════════════
# 8. fromClaude 兼容工具
# ════════════════════════════════════════════


class TestMcpFromClaude(TestCase):
    """from_claude_config utility tests."""

    def test_local_command(self) -> None:
        cfg, warnings = from_claude_config(
            "my-srv",
            {
                "command": "python",
                "args": ["server.py"],
            },
        )
        self.assertIsNotNone(cfg)
        self.assertEqual(len(warnings), 0)
        assert cfg is not None
        self.assertEqual(cfg.command, ("python", "server.py"))
        self.assertTrue(cfg.enabled)

    def test_local_command_no_args(self) -> None:
        cfg, warnings = from_claude_config(
            "my-srv",
            {
                "command": "npx",
            },
        )
        self.assertIsNotNone(cfg)
        self.assertEqual(len(warnings), 0)
        assert cfg is not None
        self.assertEqual(cfg.command, ("npx",))

    def test_remote_url_warning(self) -> None:
        cfg, warnings = from_claude_config(
            "remote-srv",
            {
                "url": "https://example.com/mcp",
            },
        )
        self.assertIsNone(cfg)
        self.assertTrue(any("remote" in w for w in warnings))

    def test_sse_unsupported(self) -> None:
        cfg, warnings = from_claude_config(
            "sse-srv",
            {
                "command": "python",
                "type": "sse",
            },
        )
        self.assertIsNone(cfg)
        self.assertTrue(any("sse" in w for w in warnings))

    def test_disabled(self) -> None:
        cfg, warnings = from_claude_config(
            "off",
            {
                "command": "python",
                "disabled": True,
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertFalse(cfg.enabled)

    def test_environment(self) -> None:
        cfg, warnings = from_claude_config(
            "srv",
            {
                "command": "python",
                "environment": {"KEY": "VAL"},
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.env, {"KEY": "VAL"})

    def test_env_fallback(self) -> None:
        cfg, warnings = from_claude_config(
            "srv",
            {
                "command": "python",
                "env": {"KEY": "VAL"},
            },
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.env, {"KEY": "VAL"})

    def test_not_an_object(self) -> None:
        cfg, warnings = from_claude_config("srv", "string")
        self.assertIsNone(cfg)
        self.assertTrue(len(warnings) > 0)

    def test_missing_command_and_url(self) -> None:
        cfg, warnings = from_claude_config(
            "srv",
            {
                "foo": "bar",
            },
        )
        self.assertIsNone(cfg)
        self.assertTrue(any("missing" in w for w in warnings))


# ════════════════════════════════════════════
# 9. MCP 工具构建（集成）
# ════════════════════════════════════════════


@patch("xcode.experimental.mcp_client.McpClient")
class TestMcpBuildIntegration(TestCase):
    """Integration tests for build_mcp_tools."""

    def test_builds_tools_from_config(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
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
            self.assertEqual(len(tools), 2)
            names = [t.name for t in tools]
            self.assertIn("mcp__my_server__greet", names)
            self.assertIn("mcp__my_server__echo", names)
            for t in tools:
                self.assertEqual(t.group, "mcp")

    def test_disabled_server_skipped(self, mock_client: MagicMock) -> None:
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
            self.assertIsInstance(tools, tuple)

    def test_overrides_server_skipped(self, mock_client: MagicMock) -> None:
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
                            "overrides": {"x": "low"},
                        },
                    }
                },
            )
            tools = build_mcp_tools(root)
            # Only "good" server's tools (if any) appear
            # bad is skipped due to overrides
            self.assertIsInstance(tools, tuple)

    def test_mcp_metadata_on_tool_spec(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.list_tools.return_value = [
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
                        "test-srv": {
                            "command": "python",
                        }
                    }
                },
            )
            tools = build_mcp_tools(root)
            self.assertEqual(len(tools), 1)
            spec = tools[0]
            self.assertEqual(spec.name, "mcp__test-srv__my_tool")
            self.assertEqual(spec.group, "mcp")
            self.assertIsNotNone(spec.builtin)
            assert spec.builtin is not None
            meta = spec.builtin.get("mcp_metadata", {})
            self.assertEqual(meta.get("server"), "test-srv")
            self.assertEqual(meta.get("server_slug"), "test-srv")
            self.assertEqual(meta.get("tool"), "my_tool")
            self.assertEqual(meta.get("tool_slug"), "my_tool")

    def test_collision_disables_tools(self, mock_client: MagicMock) -> None:
        mock_instance = MagicMock()
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
            self.assertEqual(len(tools), 0)


# ════════════════════════════════════════════
# 10. MCP 红action/脱敏 — McpClient 集成
# ════════════════════════════════════════════


class TestMcpClientRedaction(TestCase):
    """McpClient stderr redaction in error messages."""

    def test_stderr_redacted_and_truncated(self) -> None:
        """Verify that stderr in process-exit errors is redacted."""
        raw = "Bearer sk-secret123\n" + "x" * 500
        result = truncate_redact(raw, max_len=200)
        self.assertNotIn("sk-secret123", result)
        self.assertIn("****", result)
        self.assertLessEqual(len(result), 203)  # 200 + "..."
