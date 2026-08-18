"""bash 工具请求解析纯函数单元测试。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from xcode.agent.types import TerminalRenderIntent
from xcode.coding_agent.tools.bash import (
    build_bash_tool,
    _parse_bash_request,
    _parse_timeout,
    _parse_workdir,
)
from xcode.coding_agent.tools.shell_adapter import ShellSpec
from xcode.harness.execution_env import ExecutionResult


class _RecordingShell:
    def __init__(self) -> None:
        self.argv: list[str] = []
        self.cwd: Path | None = None
        self.timeout = 0

    def run(
        self,
        argv: list[str],
        cwd: Path,
        timeout: int = 30_000,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        del cancel_event, env
        self.argv = argv
        self.cwd = cwd
        self.timeout = timeout
        if on_progress is not None:
            on_progress("local output")
        return ExecutionResult(stdout="local output", stderr="", returncode=0)


class TestParseBashRequest:
    def test_valid(self) -> None:
        request = _parse_bash_request({"command": "echo hi"})
        assert request.command == "echo hi"
        assert request.timeout > 0

    def test_missing_command_raises(self) -> None:
        with pytest.raises(ValueError, match="command"):
            _parse_bash_request({})

    def test_input_alias_is_not_accepted(self) -> None:
        with pytest.raises(ValueError, match="command"):
            _parse_bash_request({"input": "echo legacy"})


class TestParseTimeout:
    def test_default(self) -> None:
        assert _parse_timeout({}) == 30000

    def test_timeout_ms(self) -> None:
        assert _parse_timeout({"timeout_ms": 5000}) == 5000

    def test_seconds_parameter_is_not_accepted(self) -> None:
        with pytest.raises(ValueError, match="unsupported bash parameter"):
            _parse_timeout({"timeout": 60})

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _parse_timeout({"timeout_ms": -1})

    def test_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="<= 300000"):
            _parse_timeout({"timeout_ms": 999999})

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_timeout({"timeout_ms": "abc"})


class TestParseWorkdir:
    def test_none(self) -> None:
        assert _parse_workdir({}) is None

    def test_valid(self) -> None:
        assert _parse_workdir({"workdir": "src"}) == "src"

    def test_empty_string(self) -> None:
        assert _parse_workdir({"workdir": ""}) is None


def test_bash_tool_depends_directly_on_local_shell(tmp_path: Path) -> None:
    shell = _RecordingShell()
    tool = build_bash_tool(
        tmp_path,
        shell_spec=ShellSpec("sh", ("sh", "-c"), "posix"),
        shell=shell,
    )

    output = tool.handler(
        {"command": "printf local", "timeout_ms": 1234},
        None,
    )

    assert output == "local output"
    assert shell.argv == ["sh", "-c", "printf local"]
    assert shell.cwd == tmp_path.resolve()
    assert shell.timeout == 1234
    assert "timeout" not in (tool.schema or {})["properties"]
    assert output.render_intent == TerminalRenderIntent(
        command="printf local",
        cwd=tmp_path.resolve().as_posix(),
    )
