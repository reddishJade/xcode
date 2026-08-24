"""进程 sandbox 契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from xcode.coding_agent.assembly.security import sandbox_policy_from_security
from xcode.harness.config import SecurityRuntimeConfig
from xcode.harness.execution_env import (
    LinuxBubblewrapSandbox,
    NetworkAccess,
    SandboxedCommand,
    SandboxMode,
    SandboxPolicy,
    SandboxUnavailableError,
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


class _ViolatingSandbox:
    def wrap(self, argv: list[str], cwd: Path) -> SandboxedCommand:
        del argv
        return SandboxedCommand(
            argv=("true",),
            cwd=cwd,
            finalize=lambda: "sandbox policy violation",
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


def test_subprocess_shell_reports_finalize_violation(tmp_path: Path) -> None:
    updates: list[str] = []

    result = SubprocessShell(sandbox=_ViolatingSandbox()).run(
        ["true"],
        tmp_path,
        on_progress=updates.append,
    )

    assert result.returncode == 126
    assert result.stderr == "sandbox policy violation"
    assert "sandbox policy violation" in "".join(updates)


def _fake_bwrap(tmp_path: Path) -> Path:
    executable = tmp_path / "bwrap"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def test_linux_workspace_write_wraps_command_and_protects_metadata(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in (".git", ".agents", ".xcode"):
        (project / name).mkdir()
    external = tmp_path / "external"
    external.mkdir()
    bwrap = _fake_bwrap(tmp_path)
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(
            project_root=project,
            writable_roots=(external,),
        ),
        bwrap_path=bwrap,
    )

    command = sandbox.wrap(["sh", "-c", "printf ok"], project)

    assert command.cwd == project
    assert command.argv[0] == str(bwrap)
    assert command.argv[-3:] == ("sh", "-c", "printf ok")
    assert ("--ro-bind", "/", "/") == command.argv[3:6]
    assert "--unshare-user" in command.argv
    assert "--unshare-pid" in command.argv
    assert "--unshare-ipc" in command.argv
    assert "--unshare-net" in command.argv
    assert _mounts(command.argv, "--bind") == {
        (str(project), str(project)),
        (str(external), str(external)),
    }
    assert _mounts(command.argv, "--ro-bind") >= {
        ("/", "/"),
        *(
            (str(project / name), str(project / name))
            for name in (
                ".git",
                ".agents",
                ".xcode",
            )
        ),
    }


def test_linux_network_allow_keeps_host_network(tmp_path: Path) -> None:
    bwrap = _fake_bwrap(tmp_path)
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(
            project_root=tmp_path,
            network_access=NetworkAccess.ALLOW,
        ),
        bwrap_path=bwrap,
    )

    command = sandbox.wrap(["true"], tmp_path)

    assert "--unshare-net" not in command.argv


def test_linux_masks_unreadable_files_and_directories(tmp_path: Path) -> None:
    secret_file = tmp_path / ".env"
    secret_file.write_text("TOKEN=secret", encoding="utf-8")
    secret_directory = tmp_path / ".ssh"
    secret_directory.mkdir()
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(
            project_root=tmp_path,
            unreadable_roots=(secret_file, secret_directory),
        ),
        bwrap_path=_fake_bwrap(tmp_path),
    )

    command = sandbox.wrap(["true"], tmp_path)

    assert ("/dev/null", str(secret_file)) in _mounts(command.argv, "--ro-bind")
    assert ("--tmpfs", str(secret_directory)) in tuple(
        zip(command.argv, command.argv[1:])
    )


def test_linux_read_only_does_not_bind_project_writable(tmp_path: Path) -> None:
    bwrap = _fake_bwrap(tmp_path)
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(project_root=tmp_path, mode=SandboxMode.READ_ONLY),
        bwrap_path=bwrap,
    )

    command = sandbox.wrap(["true"], tmp_path)

    assert _mounts(command.argv, "--bind") == set()


def test_linux_rejects_command_cwd_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(project_root=project),
        bwrap_path=_fake_bwrap(tmp_path),
    )

    with pytest.raises(ValueError, match="cwd must stay inside"):
        sandbox.wrap(["true"], outside)


def test_linux_missing_bwrap_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(
        SandboxUnavailableError, match="bubblewrap executable not found"
    ):
        LinuxBubblewrapSandbox(SandboxPolicy(project_root=tmp_path))


def test_linux_protects_missing_xcode_metadata_with_placeholder(
    tmp_path: Path,
) -> None:
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(project_root=tmp_path),
        bwrap_path=_fake_bwrap(tmp_path),
    )
    command = sandbox.wrap(["true"], tmp_path)
    metadata = tmp_path / ".xcode"

    assert metadata.is_dir()
    assert (str(metadata), str(metadata)) in _mounts(command.argv, "--ro-bind")
    assert command.finalize is not None
    violation = command.finalize()

    assert violation is None
    assert not metadata.exists()


def test_linux_placeholder_cleanup_preserves_replacement(tmp_path: Path) -> None:
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(project_root=tmp_path),
        bwrap_path=_fake_bwrap(tmp_path),
    )
    command = sandbox.wrap(["true"], tmp_path)
    metadata = tmp_path / ".xcode"
    metadata.rmdir()
    metadata.mkdir()
    settings = metadata / "settings.json"
    settings.write_text("user data", encoding="utf-8")

    assert command.finalize is not None
    violation = command.finalize()

    assert violation is not None
    assert "identity changed" in violation
    assert settings.read_text(encoding="utf-8") == "user data"


def test_linux_rejects_protected_metadata_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / ".xcode").symlink_to(target, target_is_directory=True)
    sandbox = LinuxBubblewrapSandbox(
        SandboxPolicy(project_root=tmp_path),
        bwrap_path=_fake_bwrap(tmp_path),
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        sandbox.wrap(["true"], tmp_path)


def _mounts(argv: tuple[str, ...], flag: str) -> set[tuple[str, str]]:
    mounts: set[tuple[str, str]] = set()
    for index, token in enumerate(argv):
        if token == flag:
            mounts.add((argv[index + 1], argv[index + 2]))
    return mounts


def test_runtime_security_maps_writable_external_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    writable = tmp_path / "writable"
    writable.mkdir()
    readable = tmp_path / "readable"
    readable.mkdir()
    security = SecurityRuntimeConfig.model_validate(
        {
            "external_directories": [
                {"path": str(writable), "access": "read_write"},
                {"path": str(readable), "access": "read"},
            ]
        }
    )

    policy = sandbox_policy_from_security(project, security)

    assert writable in policy.writable_roots
    assert readable not in policy.writable_roots


def test_runtime_security_masks_project_credentials(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = project / ".env"
    environment.write_text("TOKEN=secret", encoding="utf-8")
    environment_example = project / ".env.example"
    environment_example.write_text("TOKEN=", encoding="utf-8")
    credentials = project / "service" / ".aws"
    credentials.mkdir(parents=True)
    (credentials / "credentials").write_text("secret", encoding="utf-8")

    policy = sandbox_policy_from_security(project, SecurityRuntimeConfig())

    assert environment in policy.unreadable_roots
    assert environment_example not in policy.unreadable_roots
    assert credentials in policy.unreadable_roots


def test_runtime_security_honors_environment_read_override(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = project / ".env"
    environment.write_text("TOKEN=allowed", encoding="utf-8")
    security = SecurityRuntimeConfig.model_validate(
        {
            "sensitive_path_overrides": [
                {"path": ".env", "access": "read"},
            ]
        }
    )

    policy = sandbox_policy_from_security(project, security)

    assert environment not in policy.unreadable_roots
