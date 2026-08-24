"""进程 sandbox 契约测试。"""

from __future__ import annotations

from pathlib import Path

from xcode.harness.execution_env import (
    SandboxedCommand,
    SandboxMode,
    SandboxPolicy,
    SubprocessShell,
)


class _RecordingSandbox:
    def __init__(self) -> None:
        self.received: tuple[tuple[str, ...], Path] | None = None

    def wrap(self, argv: list[str], cwd: Path) -> SandboxedCommand:
        self.received = (tuple(argv), cwd)
        return SandboxedCommand(
            argv=("sh", "-c", "printf sandboxed"),
            cwd=cwd,
        )


def test_default_policy_protects_xcode_metadata(tmp_path: Path) -> None:
    policy = SandboxPolicy(project_root=tmp_path)

    assert policy.mode is SandboxMode.WORKSPACE_WRITE
    assert policy.protected_workspace_paths == (".git", ".agents", ".xcode")


def test_subprocess_shell_applies_sandbox_wrapper(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    shell = SubprocessShell(sandbox=sandbox)

    result = shell.run(["sh", "-c", "printf original"], tmp_path)

    assert result.returncode == 0
    assert result.stdout == "sandboxed"
    assert sandbox.received == (("sh", "-c", "printf original"), tmp_path)
