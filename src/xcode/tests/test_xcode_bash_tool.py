from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xcode.cli.repl_tools import parse_tool_input
from xcode.harness.tools import build_bash_tool
from xcode.harness.skills import run_tool


class XcodeBashToolTests(unittest.TestCase):
    def test_bash_safe_command_does_not_require_hitl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = run_tool(
                {tool.name: tool}, "bash", {"command": "git status --short"}
            )
            self.assertNotIn("需要授权", output)

    def test_bash_runs_command_and_returns_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "echo hello"})
            self.assertIn("hello", output)

    def test_bash_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler(
                {"command": 'python -c "import time; time.sleep(5)"', "timeout": 1}
            )
            self.assertIn("timed out", output)

    def test_bash_returns_exit_code_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "exit 1"})
            self.assertIn("exit code: 1", output)

    def test_bash_runs_structured_tool_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = run_tool(
                {tool.name: tool},
                "bash",
                {"command": "echo hello", "timeout": 5},
            )
            self.assertIn("hello", output)

    def test_bash_rejects_invalid_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            with self.assertRaisesRegex(ValueError, "invalid JSON input"):
                parse_tool_input(tool, '{"command": "echo hello",')

    def test_bash_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            with self.assertRaisesRegex(ValueError, "JSON input must be an object"):
                parse_tool_input(tool, '["echo hello"]')

    def test_bash_cli_shorthand_uses_schema_required_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            self.assertEqual(
                parse_tool_input(tool, "echo hello"),
                {"command": "echo hello"},
            )

    def test_bash_rejects_invalid_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            with self.assertRaisesRegex(ValueError, "timeout must be an integer"):
                tool.handler({"command": "echo hello", "timeout": "bad"})
            with self.assertRaisesRegex(ValueError, "timeout must be positive"):
                tool.handler({"command": "echo hello", "timeout": 0})

    def test_bash_tool_has_structured_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            self.assertIsNotNone(tool.schema)
            assert tool.schema is not None
            self.assertEqual(tool.schema["required"], ["command"])
            self.assertIn("timeout", tool.schema["properties"])

    def test_bash_static_risk_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            self.assertEqual(tool.risk, "low")


if __name__ == "__main__":
    unittest.main()
