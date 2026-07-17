"""权限模型工具函数单元测试。"""

from __future__ import annotations

from xcode.harness.security.permission_model.utils import (
    _command_grant_pattern,
    _grant_target_pattern,
    _looks_absolute,
    _is_sensitive_path,
    _is_blocked_workspace_path,
    _is_git_path,
    _access_satisfies,
    create_grant_record,
)
from xcode.harness.security.permission_model.types import (
    Action,
    Target,
    GrantRecord,
)


class TestCommandGrantPattern:
    def test_simple_command(self) -> None:
        result = _command_grant_pattern("ls -la")
        assert result == "ls *"

    def test_git_subcommand(self) -> None:
        result = _command_grant_pattern("git push origin main")
        assert result == "git push *"

    def test_npm_run(self) -> None:
        result = _command_grant_pattern("npm run build --production")
        assert result == "npm run build *"

    def test_complex_shlex(self) -> None:
        result = _command_grant_pattern("echo 'hello world' > /dev/null")
        assert result.startswith("echo")


class TestGrantTargetPattern:
    def test_command_kind(self) -> None:
        target = Target(kind="command", value="ls -la", access="execute")
        result = _grant_target_pattern(target)
        assert result == "ls *"

    def test_path_kind(self) -> None:
        target = Target(kind="path", value="/tmp/file.txt", access="read")
        result = _grant_target_pattern(target)
        assert result == "/tmp/file.txt"


class TestLooksAbsolute:
    def test_unix_path(self) -> None:
        assert _looks_absolute("/etc/passwd")

    def test_windows_path(self) -> None:
        assert _looks_absolute("C:/Users/name")

    def test_relative(self) -> None:
        assert not _looks_absolute("src/main.py")

    def test_empty(self) -> None:
        assert not _looks_absolute("")


class TestIsSensitivePath:
    def test_dotenv(self) -> None:
        assert _is_sensitive_path(".env")
        assert _is_sensitive_path(".env.production")

    def test_dotenv_example_write_only(self) -> None:
        assert _is_sensitive_path(".env.example", access="write")
        assert not _is_sensitive_path(".env.example", access="read")

    def test_credential_paths(self) -> None:
        assert _is_sensitive_path(".ssh/id_rsa")
        assert _is_sensitive_path(".aws/config")

    def normal_path_not_sensitive(self) -> None:
        assert not _is_sensitive_path("src/main.py")


class TestIsBlockedWorkspacePath:
    def test_venv(self) -> None:
        assert _is_blocked_workspace_path(".venv/lib/python")

    def test_pycache(self) -> None:
        assert _is_blocked_workspace_path("src/__pycache__/foo.pyc")

    def test_chroma_db(self) -> None:
        assert _is_blocked_workspace_path(".local/chroma_db/data")

    def test_normal_path(self) -> None:
        assert not _is_blocked_workspace_path("src/main.py")


class TestIsGitPath:
    def test_dotgit(self) -> None:
        assert _is_git_path(".git/config")

    def test_non_git(self) -> None:
        assert not _is_git_path("src/main.py")


class TestAccessSatisfies:
    def test_read_write_covers_all(self) -> None:
        assert _access_satisfies("read_write", "read")
        assert _access_satisfies("read_write", "write")

    def test_read_only_read(self) -> None:
        assert _access_satisfies("read", "read")
        assert not _access_satisfies("read", "write")

    def test_write_only_write(self) -> None:
        assert _access_satisfies("write", "write")
        assert not _access_satisfies("write", "read")


class TestCreateGrantRecord:
    def test_basic(self) -> None:
        action = Action(
            tool="read_file",
            capability="read",
            operation="read_file",
            targets=(),
            input={},
        )
        target = Target(kind="path", value="src/main.py", access="read")
        record = create_grant_record(action, target, decision="allow", scope="session")
        assert isinstance(record, GrantRecord)
        assert record.capability == "read"
        assert record.decision == "allow"
        assert record.scope == "session"
