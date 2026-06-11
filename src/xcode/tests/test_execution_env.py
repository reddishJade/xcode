from __future__ import annotations

import sys
import threading
from pathlib import Path
import tempfile
import unittest

from xcode.harness.execution_env import (
    ExecutionResult,
    MockExecutionEnv,
    SubprocessExecutionEnv,
)
from xcode.coding_agent.tools import build_bash_tool


class TestSubprocessExecutionEnv(unittest.TestCase):
    def test_runs_command_and_returns_output(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            [sys.executable, "-c", 'import sys; sys.stdout.write("hello\\n")'],
            cwd=Path("/"),
        )
        self.assertEqual(result.stdout.strip(), "hello")
        self.assertEqual(result.returncode, 0)

    def test_returns_stderr(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            [sys.executable, "-c", 'import sys; sys.stderr.write("err")'],
            cwd=Path("/"),
        )
        self.assertEqual(result.stderr.strip(), "err")

    def test_timeout_kills_process(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
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
                [sys.executable, "-c", "import time; time.sleep(5)"],
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


if __name__ == "__main__":
    unittest.main()
