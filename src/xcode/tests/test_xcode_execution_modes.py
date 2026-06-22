from __future__ import annotations

from unittest.mock import Mock, patch

from xcode.ai.events import ToolCall
from xcode.cli.repl_hitl import ReplHITLHandler
from xcode.harness.agent_runtime.execution_modes import ActPolicy, BuildPolicy
from xcode.harness.skills import ToolSpec
import pytest


class ExecutionModeTests:
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

    def test_build_policy_denies_network_tool(self) -> None:
        policy = BuildPolicy()
        result = policy.check_call(
            ToolCall(id="t1", name="curl", input={"url": "http://example.com"})
        )
        assert result == "deny"


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
