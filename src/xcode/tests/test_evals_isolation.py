"""bubblewrap mount 边界契约测试。"""

from pathlib import Path

from xcode.evals.isolation import BubblewrapExecutor


def test_bubblewrap_mounts_runtime_but_not_control_repository(tmp_path: Path) -> None:
    source = tmp_path / "source/xcode"
    virtualenv = tmp_path / "venv"
    runtime = tmp_path / "python"
    uv_executable = tmp_path / "uv"
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    for path in (source, virtualenv, runtime, workspace, output):
        path.mkdir(parents=True)
    uv_executable.touch()
    executor = BubblewrapExecutor(
        xcode_source=source,
        virtualenv=virtualenv,
        python_runtime=runtime,
        uv_executable=uv_executable,
    )

    command = executor._command(workspace=workspace, output=output)

    assert "--clearenv" in command
    assert str(source.resolve()) in command
    assert str(workspace.resolve()) in command
    assert str(output.resolve()) in command
    assert "/runtime/xcode" in command
    assert "/run/systemd/resolve" in command
    assert command[command.index("VIRTUAL_ENV") + 1] == str(virtualenv.resolve())
    assert str(uv_executable.resolve()) in command
    assert f"/tools:{virtualenv.resolve()}/bin:/usr/bin:/bin" in command
    assert "evals/private" not in " ".join(command)
    assert "xcode.config.json" not in " ".join(command)


def test_bubblewrap_accepts_precreated_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    output.mkdir(parents=True, exist_ok=True)

    assert output.is_dir()
