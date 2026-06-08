from __future__ import annotations

import threading
from pathlib import Path
import tempfile
from typing import cast
import unittest

from xcode.harness.execution_env import (
    ExecutionResult,
    MockExecutionEnv,
    SubprocessExecutionEnv,
)
from xcode.agent.types import ShellCallOutputContent
from xcode.coding_agent.tools import ShellSpec, build_bash_tool, build_native_shell_tool
from xcode.harness.skills import AGENT_CONTENT_BLOCKS_METADATA_KEY, ToolOutput


class TestSubprocessExecutionEnv(unittest.TestCase):
    def test_runs_command_and_returns_output(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(["echo", "hello"], cwd=Path("/"))
        self.assertEqual(result.stdout.strip(), "hello")
        self.assertEqual(result.returncode, 0)

    def test_returns_stderr(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            ["python", "-c", "import sys; sys.stderr.write('err')"],
            cwd=Path("/"),
        )
        self.assertEqual(result.stderr.strip(), "err")

    def test_timeout_kills_process(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            ["python", "-c", "import time; time.sleep(5)"],
            cwd=Path("/"),
            timeout=1,
        )
        self.assertTrue(result.timed_out)

    def test_cancellation(self) -> None:
        env = SubprocessExecutionEnv()
        evt = threading.Event()

        def cancel() -> None:
            evt.set()

        timer = threading.Timer(0.3, cancel)
        timer.start()
        try:
            result = env.run(
                ["python", "-c", "import time; time.sleep(5)"],
                cwd=Path("/"),
                timeout=10,
                cancel_event=evt,
            )
            self.assertTrue(result.cancelled)
        finally:
            timer.cancel()


class TestMockExecutionEnv(unittest.TestCase):
    def test_records_calls(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="mocked"))
        result = env.run(["echo", "hi"], cwd=Path("/tmp"), timeout=5)
        self.assertEqual(result.stdout, "mocked")
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(env.calls[0][0], ["echo", "hi"])

    def test_default_result_when_no_enqueued(self) -> None:
        env = MockExecutionEnv()
        result = env.run(["echo", "hi"], cwd=Path("/"))
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.returncode, 0)

    def test_multiple_calls_consume_in_order(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="first"))
        env.enqueue(ExecutionResult(stdout="second"))
        r1 = env.run(["cmd1"], cwd=Path("/"))
        r2 = env.run(["cmd2"], cwd=Path("/"))
        self.assertEqual(r1.stdout, "first")
        self.assertEqual(r2.stdout, "second")


class TestBashWithMockEnv(unittest.TestCase):
    def test_bash_uses_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="hello from sandbox"))

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "echo real"})
            self.assertIn("hello from sandbox", output)

    def test_bash_timeout_from_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(
            ExecutionResult(stdout="", stderr="", returncode=-1, timed_out=True)
        )

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "sleep 10", "timeout": 1})
            self.assertIn("timed out", output)

    def test_bash_cancelled_from_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(
            ExecutionResult(stdout="", stderr="", returncode=-1, cancelled=True)
        )

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "sleep 10"})
            self.assertIn("cancelled", output)

    def test_bash_exit_code_from_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="", stderr="", returncode=42))

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "exit 42"})
            self.assertIn("exit code: 42", output)


class TestNativeShellWithMockEnv(unittest.TestCase):
    def test_native_shell_returns_official_output_block(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="Python 3.11\n", stderr="", returncode=0))

        with tempfile.TemporaryDirectory() as tmp:
            shell_spec = ShellSpec("sh", ("sh", "-c"), "posix")
            tool = build_native_shell_tool(Path(tmp), shell_spec=shell_spec, env=env)
            output = cast(
                ToolOutput,
                tool.handler(
                    {
                        "commands": ["python --version"],
                        "timeout_ms": 2500,
                        "max_output_length": 4096,
                    }
                ),
            )

        self.assertEqual(env.calls[0][0], ["sh", "-c", "python --version"])
        self.assertEqual(env.calls[0][2], 3)
        blocks = output.metadata[AGENT_CONTENT_BLOCKS_METADATA_KEY]
        self.assertIsInstance(blocks[0], ShellCallOutputContent)
        block = blocks[0]
        assert isinstance(block, ShellCallOutputContent)
        self.assertEqual(block.max_output_length, 4096)
        self.assertEqual(
            block.output,
            [
                {
                    "stdout": "Python 3.11\n",
                    "stderr": "",
                    "outcome": {"type": "exit", "exit_code": 0},
                }
            ],
        )

    def test_native_shell_timeout_uses_timeout_outcome(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(returncode=-1, timed_out=True))

        with tempfile.TemporaryDirectory() as tmp:
            shell_spec = ShellSpec("sh", ("sh", "-c"), "posix")
            tool = build_native_shell_tool(Path(tmp), shell_spec=shell_spec, env=env)
            output = cast(
                ToolOutput,
                tool.handler({"commands": ["sleep 10"], "timeout_ms": 1000}),
            )

        blocks = output.metadata[AGENT_CONTENT_BLOCKS_METADATA_KEY]
        block = blocks[0]
        assert isinstance(block, ShellCallOutputContent)
        self.assertEqual(block.output[0]["outcome"], {"type": "timeout"})


if __name__ == "__main__":
    unittest.main()
