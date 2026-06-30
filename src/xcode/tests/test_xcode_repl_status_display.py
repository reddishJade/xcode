from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile

from xcode.cli.repl_settings import list_permissions
from xcode.cli.repl_settings import add_permission_rule_interactive
from xcode.cli.repl_settings import format_permission_rule
from xcode.cli.repl_settings import handle_permissions
from xcode.cli.repl_settings import print_permission_overview
from xcode.cli.repl_settings import _render_recently_denied_tab
from xcode.cli.repl_settings import _render_rule_tab
from xcode.cli.repl_settings import _render_workspace_tab
from xcode.cli.repl_tools import run_tool_command
from xcode.harness.observability import (
    FileGrantStore,
    GrantRecord,
    InMemoryGrantStore,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.skills import ToolSpec
from unittest.mock import patch


class ToolListApp:
    def __init__(self) -> None:
        self.registry = (
            ToolSpec("read_file", "Read files.", "path", lambda _value: "", group="core"),
            ToolSpec(
                "mcp__demo__echo",
                "Echo. [mcp: demo]",
                "input",
                lambda _value: "",
                group="mcp",
            ),
        )


def test_tool_list_uses_markdown_lists_for_stable_rendering() -> None:
    output = run_tool_command("/tool list", ToolListApp())

    assert "## Visible Tools (2)" in output
    assert "- **core** (1)" in output
    assert "  - `read_file`" in output
    assert "  - `mcp__demo__echo` - MCP server `demo`" in output
    assert "Visible tools core:" not in output


def test_permissions_empty_state_explains_default_policy_layers() -> None:
    output = StringIO()

    with tempfile.TemporaryDirectory() as temp_dir:
        with redirect_stdout(output):
            list_permissions(
                InMemoryGrantStore(),
                FileGrantStore(Path(temp_dir) / "approval_grants.json"),
            )

    rendered = output.getvalue()
    assert "Permission Status" in rendered
    assert "Static rules: none" in rendered
    assert "Default from config: not set" in rendered
    assert "Implicit fallback: allow" in rendered
    assert "<permissions>" not in rendered


def test_permissions_show_static_rules_and_saved_grants() -> None:
    session_store = InMemoryGrantStore(
        (
            GrantRecord(
                capability="file",
                operation="read",
                target_kind="path",
                target_pattern="src/xcode/**",
                access="read",
                decision="allow",
                scope="session",
                grant_id="grant-1",
            ),
        )
    )
    output = StringIO()

    with tempfile.TemporaryDirectory() as temp_dir:
        permanent_store = FileGrantStore(Path(temp_dir) / "approval_grants.json")
        with redirect_stdout(output):
            list_permissions(
                session_store,
                permanent_store,
                static_policy=PermissionPolicy(
                    (
                        StaticPermission(
                            tool="bash",
                            decision="ask",
                            input_prefix="git ",
                            target_type="command",
                        ),
                    ),
                    global_default="allow",
                ),
                restricted_dirs=("C:/secrets",),
            )

    rendered = output.getvalue()
    assert "Static rules (1)" in rendered
    assert "tool `bash` -> ask (prefix=git ) [command]" in rendered
    assert "Default from config: allow" in rendered
    assert "Restricted directories (1)" in rendered
    assert "Session grants (1)" in rendered
    assert "allow file/read read on path `src/xcode/**`" in rendered


def test_permission_rule_format_matches_command_templates() -> None:
    rule = {
        "tool": "bash",
        "decision": "allow",
        "input_prefix": "uv run ",
        "target_type": "command",
    }

    assert format_permission_rule(rule) == (
        "Bash(uv run *): allow target_type=command"
    )


def test_add_permission_rule_interactive_saves_custom_rule() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "xcode.config.json"
        config: dict[str, object] = {}

        with (
            patch("questionary.select") as mock_select,
            patch("questionary.text") as mock_text,
        ):
            mock_select.return_value.ask.side_effect = [
                "allow",
                "command",
                "prefix",
            ]
            mock_text.return_value.ask.side_effect = [
                "bash",
                "",
                "git add ",
            ]
            changed = add_permission_rule_interactive(config, config_path)

        assert changed
        saved = config_path.read_text(encoding="utf-8")
        assert '"decision": "allow"' in saved
        assert '"input_prefix": "git add"' in saved


def test_permission_overview_shows_panel_tabs() -> None:
    output = StringIO()
    config = {
        "security": {
            "rules": [
                {"tool": "bash", "decision": "allow"},
                {"tool": "write_file", "decision": "ask"},
                {"tool": "apply_patch", "decision": "deny"},
            ]
        }
    }

    with redirect_stdout(output):
        print_permission_overview(config, InMemoryGrantStore(), None, None)

    rendered = output.getvalue()
    assert "Permissions" in rendered
    assert "Recently denied 0" in rendered
    assert "Allow 1" in rendered
    assert "Ask 1" in rendered
    assert "Deny 1" in rendered
    assert "Workspace" in rendered


def test_permissions_default_opens_manager_in_tty() -> None:
    output = StringIO()

    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("questionary.select") as mock_select,
            redirect_stdout(output),
        ):
            mock_select.return_value.ask.return_value = "Done"
            handle_permissions(
                "/permissions",
                InMemoryGrantStore(),
                None,
                project_root=Path(temp_dir),
            )

    rendered = output.getvalue()
    assert "Permissions" in rendered
    assert "Xcode will not ask before using tools matched by Allow rules." in rendered
    assert mock_select.call_args.args[0] == "Permissions:"


def test_permission_rule_tabs_render_expected_guidance() -> None:
    output = StringIO()

    with redirect_stdout(output):
        _render_rule_tab(
            "ask",
            [
                {
                    "tool": "bash",
                    "decision": "ask",
                    "input_prefix": "uv run ",
                    "target_type": "command",
                }
            ],
        )

    rendered = output.getvalue()
    assert "Permissions  Recently denied   Allow   [Ask]   Deny   Workspace" in rendered
    assert "Xcode will always ask for confirmation" in rendered
    assert "1. Add a new rule…" in rendered
    assert "Bash(uv run *): ask" in rendered


def test_permission_recent_denials_tab_explains_empty_state() -> None:
    output = StringIO()

    with redirect_stdout(output):
        _render_recently_denied_tab(None)

    rendered = output.getvalue()
    assert "[Recently denied]" in rendered
    assert "No recent denials" in rendered
    assert "auto mode classifier" in rendered


def test_permission_workspace_tab_lists_root_and_add_directory() -> None:
    output = StringIO()

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        with redirect_stdout(output):
            _render_workspace_tab({}, root, ())

    rendered = output.getvalue()
    assert "[Workspace]" in rendered
    assert "can read files in the workspace" in rendered
    assert "Original working directory" in rendered
    assert "1. Add directory…" in rendered
