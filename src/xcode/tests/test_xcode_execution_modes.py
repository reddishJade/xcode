from __future__ import annotations

from unittest.mock import Mock, patch
from pathlib import Path

from xcode.ai.events import ToolCall
from xcode.cli.repl_hitl import ReplHITLHandler
from xcode.harness.agent_runtime.execution_modes import (
    ActPolicy,
    BuildPolicy,
    ExecutionModeState,
    PlanPolicy,
)
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.observability.permission_model import (
    ActionExtractor,
    MODE_DEFAULT_RULES,
)
from xcode.harness.observability.rule_matcher import first_match
from xcode.harness.skills import ToolSpec
import pytest


class ExecutionModeTests:

    def test_plan_policy_exposes_read_tools_and_plan_edit_only(self) -> None:
        tools = (
            ToolSpec("read_file", "Read.", "text", lambda _value: "", read_only=True),
            ToolSpec("list_dir", "List.", "text", lambda _value: "", read_only=True),
            ToolSpec("edit_file", "Edit.", "text", lambda _value: ""),
            ToolSpec("write_file", "Write.", "text", lambda _value: ""),
            ToolSpec("apply_patch", "Patch.", "text", lambda _value: ""),
            ToolSpec("bash", "Shell.", "text", lambda _value: ""),
        )

        names = {tool.name for tool in PlanPolicy().filter_tools(tools)}

        assert {"read_file", "list_dir", "write_file", "edit_file"} <= names
        assert "apply_patch" not in names
        assert "bash" not in names

    def test_build_and_act_policies_expose_all_tools(self) -> None:
        tools = (
            ToolSpec("read_file", "Read.", "text", lambda _value: "", read_only=True),
            ToolSpec("write_file", "Write.", "text", lambda _value: ""),
            ToolSpec("bash", "Shell.", "text", lambda _value: ""),
        )

        assert BuildPolicy().filter_tools(tools) == tools
        assert ActPolicy().filter_tools(tools) == tools

    def test_act_bash_still_allowed(self) -> None:
        policy = ActPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="bash", input={"command": "echo hello"})
        )
        assert result == "allow"

    def test_build_policy_allows_file_write(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(
                id="t1", name="write_file", input={"path": "foo.txt", "content": "x"}
            )
        )
        assert result == "allow"

    def test_build_policy_allows_bash(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="bash", input={"command": "echo hello"})
        )
        assert result == "allow"

    def test_build_policy_allows_any_tool_call(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="curl", input={"url": "http://example.com"})
        )
        assert result == "allow"

    def test_plan_write_rules_are_limited_to_xcode_plan_files(self) -> None:
        write_rules = tuple(
            rule
            for rule in MODE_DEFAULT_RULES["plan"]
            if rule.action in {"write_file", "edit_file"}
        )

        assert {rule.action for rule in write_rules} == {"write_file", "edit_file"}
        assert all(
            rule.resource_pattern == ".xcode/plans/*.md" for rule in write_rules
        )

    def test_act_explicitly_allows_reads_and_asks_for_writes(self) -> None:
        effects = {
            rule.action: rule.effect for rule in MODE_DEFAULT_RULES["act"]
        }

        assert effects["read_file"] == "allow"
        assert effects["write_file"] == "ask"
        assert effects["bash"] == "ask"

    def test_plan_allows_absolute_plan_path_only(self, tmp_path: Path) -> None:
        mode = ExecutionModeState()
        mode.set_mode("plan")
        gate = ToolGate(mode, None, None, None, None, "test", project_root=tmp_path)
        snapshot = gate.snapshot()
        extractor = ActionExtractor()

        plan_action = extractor.extract(
            "write_file",
            {"path": str(tmp_path / ".xcode/plans/task.md"), "content": "# Plan"},
        )
        source_action = extractor.extract(
            "write_file",
            {"path": str(tmp_path / "src/task.py"), "content": "pass"},
        )

        plan_rule = first_match(plan_action, snapshot.mode_ruleset)
        assert plan_rule is not None
        assert plan_rule.effect == "allow"
        assert first_match(source_action, snapshot.mode_ruleset) is None


class ReplHITLHandlerTests:
    def setup_method(self, method) -> None:
        self.handler = ReplHITLHandler()
        self.tool = ToolSpec("bash", "Bash.", "text", lambda _data: "")

    def test_handler_allow_once(self) -> None:
        result = self.handler._apply_choice("Allow (once)")
        assert result.decision == "allow"
        assert result.scope == "once"

    def test_handler_session_scope(self) -> None:
        result = self.handler._apply_choice("Allow this session")
        assert result.decision == "allow"
        assert result.scope == "session"

    def test_handler_permanent_scope(self) -> None:
        result = self.handler._apply_choice("Always allow")
        assert result.decision == "allow"
        assert result.scope == "permanent"

    def test_handler_deny(self) -> None:
        result = self.handler._apply_choice("Deny")
        assert result.decision == "deny"
        assert result.scope == "once"

    def test_unknown_choice_treated_as_deny(self) -> None:
        result = self.handler._apply_choice(None)
        assert result.decision == "deny"

    def test_interactive_prompt_works_inside_running_event_loop(self) -> None:
        question = Mock()
        question.ask.return_value = "Allow (once)"

        with patch("questionary.select", return_value=question):
            result = self.handler(self.tool, {"command": "rm -rf /tmp/xcode-demo"})

        assert result.decision == "allow"
        assert result.scope == "once"
        question.ask.assert_called_once()


if __name__ == "__main__":
    pytest.main()
