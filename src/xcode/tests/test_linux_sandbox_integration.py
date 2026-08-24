"""Linux bubblewrap 的真实文件和网络边界测试。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from xcode.coding_agent.assembly.security import sandbox_policy_from_security
from xcode.harness.config import SecurityRuntimeConfig
from xcode.harness.execution_env import (
    LinuxBubblewrapSandbox,
    SandboxMode,
    SandboxPolicy,
    SandboxUnavailableError,
    SubprocessShell,
)


def _usable_shell(project: Path, policy: SandboxPolicy) -> SubprocessShell:
    if sys.platform != "linux" or shutil.which("bwrap") is None:
        pytest.skip("Linux bubblewrap is not installed")
    try:
        shell = SubprocessShell(LinuxBubblewrapSandbox(policy))
    except SandboxUnavailableError as exc:
        pytest.skip(str(exc))
    probe = shell.run(["true"], project)
    if probe.returncode != 0:
        pytest.skip(f"bubblewrap cannot run in this environment: {probe.stderr[:160]}")
    return shell


def test_workspace_write_allows_project_and_blocks_outside(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    shell = _usable_shell(project, SandboxPolicy(project_root=project))

    allowed = shell.run(["sh", "-c", "printf allowed > allowed.txt"], project)
    denied_path = outside / "denied.txt"
    denied = shell.run(
        ["sh", "-c", 'printf denied > "$1"', "sh", str(denied_path)],
        project,
    )

    assert allowed.returncode == 0
    assert (project / "allowed.txt").read_text(encoding="utf-8") == "allowed"
    assert denied.returncode != 0
    assert not denied_path.exists()


def test_child_process_inherits_write_boundary(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    denied_path = tmp_path / "child-denied.txt"
    shell = _usable_shell(project, SandboxPolicy(project_root=project))

    result = shell.run(
        [
            "sh",
            "-c",
            '(printf denied > "$1")',
            "sh",
            str(denied_path),
        ],
        project,
    )

    assert result.returncode != 0
    assert not denied_path.exists()


def test_existing_and_missing_xcode_metadata_are_protected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    metadata = project / ".xcode"
    metadata.mkdir()
    settings = metadata / "settings.json"
    settings.write_text("original", encoding="utf-8")
    shell = _usable_shell(project, SandboxPolicy(project_root=project))

    existing = shell.run(
        ["sh", "-c", 'printf changed > "$1"', "sh", str(settings)],
        project,
    )

    assert existing.returncode != 0
    assert settings.read_text(encoding="utf-8") == "original"

    metadata.rename(project / ".xcode-existing")
    missing = shell.run(
        ["sh", "-c", "mkdir .xcode && printf '{}' > .xcode/settings.json"],
        project,
    )

    assert missing.returncode != 0
    assert not metadata.exists()


def test_read_only_mode_blocks_project_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shell = _usable_shell(
        project,
        SandboxPolicy(project_root=project, mode=SandboxMode.READ_ONLY),
    )

    result = shell.run(["sh", "-c", "touch denied.txt"], project)

    assert result.returncode != 0
    assert not (project / "denied.txt").exists()


def test_sensitive_files_are_unreadable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = project / ".env"
    environment.write_text("TOKEN=do-not-read", encoding="utf-8")
    policy = sandbox_policy_from_security(project, SecurityRuntimeConfig())
    shell = _usable_shell(project, policy)

    result = shell.run(["cat", str(environment)], project)

    assert "do-not-read" not in result.stdout


def test_default_network_namespace_blocks_connections(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shell = _usable_shell(project, SandboxPolicy(project_root=project))

    result = shell.run(
        [
            sys.executable,
            "-c",
            "import socket; socket.create_connection(('1.1.1.1', 80), 0.2)",
        ],
        project,
    )

    assert result.returncode != 0
