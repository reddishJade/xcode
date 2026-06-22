from __future__ import annotations

import sys
import threading
from pathlib import Path
import tempfile
from xcode.harness.execution_env import ExecutionResult, SubprocessExecutionEnv
from xcode.tests.fixtures import MockExecutionEnv
from xcode.coding_agent.tools import build_bash_tool
import pytest
class TestSubprocessExecutionEnv:
    def test_runs_command_and_returns_output(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            [sys.executable, "-c", 'import sys; sys.stdout.write("hello\\n")'],
            cwd=Path("/"),
        )
        assert result.stdout.strip() == "hello"
        assert result.returncode == 0

    def test_returns_stderr(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            [sys.executable, "-c", 'import sys; sys.stderr.write("err")'],
            cwd=Path("/"),
        )
        assert result.stderr.strip() == "err"

    def test_timeout_kills_process(self) -> None:
        env = SubprocessExecutionEnv()
        result = env.run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=Path("/"),
            timeout=1,
        )
        assert result.timed_out

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
            assert result.cancelled
        finally:
            timer.cancel()

class TestMockExecutionEnv:
    def test_records_calls(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="mocked"))
        result = env.run(["echo", "hi"], cwd=Path("/tmp"), timeout=5)
        assert result.stdout == "mocked"
        assert len(env.calls) == 1
        assert env.calls[0][0] == ["echo", "hi"]

    def test_default_result_when_no_enqueued(self) -> None:
        env = MockExecutionEnv()
        result = env.run(["echo", "hi"], cwd=Path("/"))
        assert result.stdout == ""
        assert result.returncode == 0

    def test_multiple_calls_consume_in_order(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="first"))
        env.enqueue(ExecutionResult(stdout="second"))
        r1 = env.run(["cmd1"], cwd=Path("/"))
        r2 = env.run(["cmd2"], cwd=Path("/"))
        assert r1.stdout == "first"
        assert r2.stdout == "second"

class TestBashWithMockEnv:
    def test_bash_uses_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="hello from sandbox"))

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "echo real"})
            assert "hello from sandbox" in output

    def test_bash_timeout_from_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(
            ExecutionResult(stdout="", stderr="", returncode=-1, timed_out=True)
        )

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "sleep 10", "timeout": 1})
            assert "timed out" in output

    def test_bash_cancelled_from_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(
            ExecutionResult(stdout="", stderr="", returncode=-1, cancelled=True)
        )

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "sleep 10"})
            assert "cancelled" in output

    def test_bash_exit_code_from_injected_env(self) -> None:
        env = MockExecutionEnv()
        env.enqueue(ExecutionResult(stdout="", stderr="", returncode=42))

        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), env=env)
            output = tool.handler({"command": "exit 42"})
            assert "exit code: 42" in output

if __name__ == "__main__":
    pytest.main()
