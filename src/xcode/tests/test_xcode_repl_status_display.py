from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile

from xcode.cli.repl_settings import list_permissions
from xcode.cli.repl_tools import run_tool_command
from xcode.harness.observability import (
    FileGrantStore,
    GrantRecord,
    InMemoryGrantStore,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.skills import ToolSpec


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
