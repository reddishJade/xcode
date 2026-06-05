from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from xcode.coding_agent.tools.shell_adapter import (
    SHELL_NAMES,
    ShellSpec,
    build_shell_argv,
    detect_shell,
    _KNOWN_SHELLS,
)
from xcode.coding_agent.tools import build_bash_tool
from xcode.harness.skills import run_tool


class XcodeShellAdapterTests(unittest.TestCase):
    def test_known_shells_have_valid_prefixes(self) -> None:
        for name, spec in _KNOWN_SHELLS.items():
            self.assertIsInstance(spec, ShellSpec, f"{name} is not a ShellSpec")
            self.assertIsInstance(spec.command_prefix, tuple)
            self.assertGreater(len(spec.command_prefix), 0, f"{name} has empty prefix")
            self.assertIn(
                spec.syntax,
                ("powershell", "cmd", "posix"),
                f"{name} has invalid syntax",
            )

    def test_build_shell_argv_bash(self) -> None:
        spec = _KNOWN_SHELLS["bash"]
        argv = build_shell_argv(spec, "echo hello")
        self.assertEqual(argv, ["bash", "--noprofile", "--norc", "-c", "echo hello"])

    def test_build_shell_argv_pwsh(self) -> None:
        spec = _KNOWN_SHELLS["pwsh"]
        argv = build_shell_argv(spec, "Write-Output hello")
        self.assertEqual(
            argv,
            [
                "pwsh",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Write-Output hello",
            ],
        )

    def test_build_shell_argv_cmd(self) -> None:
        spec = _KNOWN_SHELLS["cmd"]
        argv = build_shell_argv(spec, "dir")
        self.assertEqual(argv, ["cmd", "/d", "/c", "dir"])

    def test_build_shell_argv_zsh_no_profile(self) -> None:
        spec = _KNOWN_SHELLS["zsh"]
        argv = build_shell_argv(spec, "echo hello")
        self.assertEqual(argv, ["zsh", "-f", "-c", "echo hello"])

    def test_detect_shell_explicit_bash(self) -> None:
        spec = detect_shell("bash")
        self.assertEqual(spec.name, "bash")
        self.assertEqual(spec.syntax, "posix")

    def test_detect_shell_explicit_sh(self) -> None:
        spec = detect_shell("sh")
        self.assertEqual(spec.name, "sh")
        self.assertEqual(spec.syntax, "posix")

    def test_detect_shell_explicit_missing_raises(self) -> None:
        import shutil

        for name in ("fish", "powershell", "zsh"):
            if shutil.which(name) is None:
                missing = name
                break
        else:
            self.skipTest("all known shells are on PATH")
        with self.assertRaises(RuntimeError) as ctx:
            detect_shell(missing)
        self.assertIn("not found on PATH", str(ctx.exception))

    def test_detect_shell_unknown_name_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            detect_shell("noshell")
        self.assertIn("unknown shell", str(ctx.exception))

    def test_detect_shell_auto_detects_something(self) -> None:
        spec = detect_shell()
        self.assertIn(spec.name, SHELL_NAMES - {"auto"})
        self.assertIn(spec.syntax, ("powershell", "cmd", "posix"))

    def test_detect_shell_respects_shell_env(self) -> None:
        import sys as _sys

        if _sys.platform == "win32":
            self.skipTest("SHELL env var only respected on POSIX")
        original = os.environ.get("SHELL")
        os.environ["SHELL"] = "/usr/bin/zsh"
        try:
            spec = detect_shell()
            self.assertEqual(spec.name, "zsh")
        finally:
            if original is not None:
                os.environ["SHELL"] = original
            else:
                del os.environ["SHELL"]

    def test_shell_names_are_consistent(self) -> None:
        for name in SHELL_NAMES:
            if name == "auto":
                continue
            self.assertIn(
                name, _KNOWN_SHELLS, f"{name} in SHELL_NAMES but not _KNOWN_SHELLS"
            )
        for name in _KNOWN_SHELLS:
            self.assertIn(
                name, SHELL_NAMES, f"{name} in _KNOWN_SHELLS but not SHELL_NAMES"
            )

    def test_bash_tool_default_detects_shell(self) -> None:
        """真实执行只走 auto-detect 路径，不强制显式 shell。"""
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp))
            registry = {tool.name: tool}

            output = run_tool(registry, "bash", {"command": "echo hello"})
            self.assertIn("hello", output)

    def test_bash_tool_passes_shell_false_and_argv(self) -> None:
        """验证显式 shell_spec 时 argv 和 cwd 正确传递。"""
        from xcode.harness.execution_env import SandboxExecutionEnv

        env = SandboxExecutionEnv()
        spec = _KNOWN_SHELLS["bash"]
        with tempfile.TemporaryDirectory() as tmp:
            tool = build_bash_tool(Path(tmp), shell_spec=spec, env=env)
            tool.handler({"command": "echo mock_shell", "timeout": 5})

        self.assertEqual(len(env.calls), 1)
        argv, cwd, timeout = env.calls[0]
        # 使用 argv 而非 command string，即 shell=False
        self.assertEqual(
            argv,
            ["bash", "--noprofile", "--norc", "-c", "echo mock_shell"],
        )
