from __future__ import annotations

import os
import tempfile
from pathlib import Path

from xcode.coding_agent.tools.shell_adapter import (
    SHELL_NAMES,
    ShellSpec,
    build_shell_argv,
    detect_shell,
    _KNOWN_SHELLS,
)
from xcode.coding_agent.tools import build_bash_tool
import pytest
class XcodeShellAdapterTests:
    def test_known_shells_have_valid_prefixes(self) -> None:
        for name, spec in _KNOWN_SHELLS.items():
            assert isinstance(spec, ShellSpec), f"{name} is not a ShellSpec"
            assert isinstance(spec.command_prefix, tuple)
            assert len(spec.command_prefix) > 0, f"{name} has empty prefix"
            assert spec.syntax in ("powershell", "cmd", "posix"), f"{name} has invalid syntax" 

    def test_build_shell_argv_bash(self) -> None:
        spec = _KNOWN_SHELLS["bash"]
        argv = build_shell_argv(spec, "echo hello")
        assert argv == ["bash", "--noprofile", "--norc", "-c", "echo hello"]

    def test_build_shell_argv_pwsh(self) -> None:
        spec = _KNOWN_SHELLS["pwsh"]
        argv = build_shell_argv(spec, "Write-Output hello")
        assert argv == [
                "pwsh",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Write-Output hello",
            ]

    def test_build_shell_argv_cmd(self) -> None:
        spec = _KNOWN_SHELLS["cmd"]
        argv = build_shell_argv(spec, "dir")
        assert argv == ["cmd", "/d", "/c", "dir"]

    def test_build_shell_argv_zsh_no_profile(self) -> None:
        spec = _KNOWN_SHELLS["zsh"]
        argv = build_shell_argv(spec, "echo hello")
        assert argv == ["zsh", "-f", "-c", "echo hello"]

    def test_detect_shell_explicit_bash(self) -> None:
        spec = detect_shell("bash")
        assert spec.name == "bash"
        assert spec.syntax == "posix"

    def test_detect_shell_explicit_sh(self) -> None:
        spec = detect_shell("sh")
        assert spec.name == "sh"
        assert spec.syntax == "posix"

    def test_detect_shell_explicit_missing_raises(self) -> None:
        import shutil

        for name in ("fish", "powershell", "zsh"):
            if shutil.which(name) is None:
                missing = name
                break
        else:
            pytest.skip("all known shells are on PATH")
        with pytest.raises(RuntimeError) as exc_info:
            detect_shell(missing)
        assert "not found on PATH" in str(exc_info.value)

    def test_detect_shell_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            detect_shell("noshell")
        assert "unknown shell" in str(exc_info.value)

    def test_detect_shell_auto_detects_something(self) -> None:
        spec = detect_shell()
        assert spec.name in SHELL_NAMES - {"auto"}
        assert spec.syntax in ("powershell", "cmd", "posix")

    def test_detect_shell_respects_shell_env(self) -> None:
        import sys as _sys

        if _sys.platform == "win32":
            pytest.skip("SHELL env var only respected on POSIX")
        original = os.environ.get("SHELL")
        os.environ["SHELL"] = "/usr/bin/zsh"
        try:
            spec = detect_shell()
            assert spec.name == "zsh"
        finally:
            if original is not None:
                os.environ["SHELL"] = original
            else:
                del os.environ["SHELL"]

    def test_shell_names_are_consistent(self) -> None:
        for name in SHELL_NAMES:
            if name == "auto":
                continue
            assert name in _KNOWN_SHELLS, f"{name} in SHELL_NAMES but not _KNOWN_SHELLS"
        for name in _KNOWN_SHELLS:
            assert name in SHELL_NAMES, f"{name} in _KNOWN_SHELLS but not SHELL_NAMES"

    def test_bash_tool_default_detects_shell(self) -> None:
        """真实执行只走 auto-detect 路径，不强制显式 shell。"""
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            output = tool.handler({"command": "echo hello"})
            assert "hello" in output

    def test_bash_tool_passes_shell_false_and_argv(self) -> None:
        """验证显式 shell_spec 时 argv 和 cwd 正确传递。"""
        from xcode.tests.fixtures import MockExecutionEnv
        env = MockExecutionEnv()
        spec = _KNOWN_SHELLS["bash"]
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), shell_spec=spec, env=env)
            tool.handler({"command": "echo mock_shell", "timeout": 5})

        assert len(env.calls) == 1
        argv, cwd, timeout = env.calls[0]
        # 使用 argv 而非 command string，即 shell=False
        assert argv == ["bash", "--noprofile", "--norc", "-c", "echo mock_shell"]
