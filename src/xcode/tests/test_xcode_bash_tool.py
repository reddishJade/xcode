from __future__ import annotations

from pathlib import Path
import tempfile
from xcode.cli.repl_tools import parse_tool_input
from xcode.coding_agent.tools import build_bash_tool
from xcode.coding_agent.tools.bash import OutputAccumulator
from xcode.harness.observability._safety_backstop import SafetyBackstopPolicyEvaluator
from xcode.harness.observability.permission_model import ActionExtractor
import pytest


class XcodeBashToolTests:
    def test_bash_safe_command_does_not_require_hitl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "git status --short"})
            assert "requires approval" not in output

    def test_bash_runs_command_and_returns_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "echo hello"})
            assert "hello" in output

    def test_bash_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler(
                {"command": 'python -c "import time; time.sleep(5)"', "timeout": 1}
            )
            assert "timed out" in output

    def test_bash_returns_exit_code_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "exit 1"})
            assert "exit code: 1" in output

    def test_bash_runs_structured_tool_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "echo hello", "timeout": 5})
            assert "hello" in output

    def test_bash_rejects_invalid_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            with pytest.raises(ValueError, match="invalid JSON input"):
                parse_tool_input(tool, '{"command": "echo hello",')

    def test_bash_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            with pytest.raises(ValueError, match="JSON input must be an object"):
                parse_tool_input(tool, '["echo hello"]')

    def test_bash_cli_shorthand_uses_schema_required_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            assert parse_tool_input(tool, "echo hello") == {"command": "echo hello"}

    def test_bash_rejects_invalid_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            with pytest.raises(ValueError, match="timeout must be an integer"):
                tool.handler({"command": "echo hello", "timeout": "bad"})
            with pytest.raises(ValueError, match="timeout must be positive"):
                tool.handler({"command": "echo hello", "timeout": 0})

    def test_bash_tool_has_structured_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            assert tool.schema is not None
            assert tool.schema is not None
            assert tool.schema["required"] == ["command"]
            assert "timeout" in tool.schema["properties"]

    def test_bash_accumulator_preserves_full_output_file(self) -> None:
        acc = OutputAccumulator(max_bytes=80, max_lines=2)
        acc.append(b"one\n")
        acc.append(b"two\n")
        acc.append(b"three\n")

        output = acc.snapshot()
        marker = "Full output: "
        assert marker in output
        full_path = output.split(marker, 1)[1].rstrip("]")
        acc.close()

        try:
            assert Path(full_path).read_text(encoding="utf-8") == "one\ntwo\nthree\n"
        finally:
            Path(full_path).unlink(missing_ok=True)

    def test_safety_backstop_covers_old_classification(self) -> None:
        """SafetyBackstopPolicy 覆盖旧的 evaluate_command_risk 分类。"""
        evaluator = SafetyBackstopPolicyEvaluator()

        def _action(cmd: str):
            return ActionExtractor().extract("bash", {"command": cmd})

        # rm 非根路径 → ask (Bucket B)
        for cmd in ["rm -rf ./tmp", "rm -rf /tmp/xcode-demo"]:
            constraints = evaluator.evaluate(_action(cmd))
            assert constraints[0].decision == "ask", cmd

        # rm 根路径 → deny non-bypassable (Bucket A)
        for cmd in ["rm -rf /", "rm -fr /*"]:
            constraints = evaluator.evaluate(_action(cmd))
            assert constraints[0].decision == "deny", cmd
            assert constraints[0].non_bypassable

        # 系统路径递归修改 → deny non-bypassable (Bucket A)
        for cmd in ["rm -rf /etc", "chmod -R 777 /usr", "chown -R root /var"]:
            constraints = evaluator.evaluate(_action(cmd))
            assert constraints[0].decision == "deny", cmd

        # 非系统路径递归权限变更 → ask (Bucket B)
        for cmd in ["chmod -R 777 ./tmp", "chown -R root ~/xcode/tmp"]:
            constraints = evaluator.evaluate(_action(cmd))
            assert constraints[0].decision == "ask", cmd


if __name__ == "__main__":
    pytest.main()
