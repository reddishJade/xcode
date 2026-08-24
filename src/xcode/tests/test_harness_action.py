"""Action 提取器纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.security.permission_model.action import (
    ActionExtractor,
    _filesystem_command_path_arguments,
    _normalize_path_text,
)


def test_read_file_action() -> None:
    extractor = ActionExtractor()
    action = extractor.extract("read_file", {"path": "src/main.py"}, ("read", "path"))
    assert action.operation == "read_file"
    assert action.capability == "read"
    assert len(action.targets) == 1
    assert action.targets[0].value == "src/main.py"


def test_write_file_action() -> None:
    extractor = ActionExtractor()
    action = extractor.extract("write_file", {"path": "src/new.py"}, ("write", "path"))
    assert action.operation == "write_file"
    assert action.targets[0].access == "write"


def test_bash_action() -> None:
    extractor = ActionExtractor()
    action = extractor.extract("bash", {"command": "ls -la"}, ("shell", "none"))
    assert action.operation == "run_command"
    assert action.capability == "shell"


def test_unknown_tool() -> None:
    extractor = ActionExtractor()
    action = extractor.extract("unknown_tool", {}, None)
    assert action.capability == "unknown"


def test_custom_tool_profile_extracts_declared_path_target() -> None:
    extractor = ActionExtractor()
    action = extractor.extract(
        "custom_export",
        {"destination": "reports/result.txt"},
        ("write", "path"),
        path_extractor=lambda data: (str(data["destination"]),),
    )

    assert action.capability == "write"
    assert action.operation == "custom_export"
    assert [target.value for target in action.targets] == ["reports/result.txt"]
    assert action.targets[0].access == "write"


def test_apply_patch_uses_injected_path_extractor() -> None:
    extractor = ActionExtractor()
    action = extractor.extract(
        "apply_patch",
        {"patch_text": "opaque"},
        ("patch", "path"),
        path_extractor=lambda _tool_input: ("src/old.py", "src/new.py"),
    )
    assert [target.value for target in action.targets] == [
        "src/old.py",
        "src/new.py",
    ]


def test_load_skill_action() -> None:
    extractor = ActionExtractor()
    action = extractor.extract("load_skill", {"name": "my-skill"}, ("skill", "skill"))
    assert action.capability == "skill"
    assert len(action.targets) == 1
    assert action.targets[0].value == "my-skill"


class TestNormalizePathText:
    def test_strips_whitespace(self) -> None:
        assert _normalize_path_text("  src/main.py  ") == "src/main.py"

    def test_absolute_path(self) -> None:
        result = _normalize_path_text("/home/user/project/src/main.py")
        assert result.startswith("/") or ":" in result

    def test_dots_removed(self) -> None:
        assert _normalize_path_text("./src/./main.py") == "src/main.py"

    def test_empty_fallsback(self) -> None:
        assert _normalize_path_text("") == "."


class TestFilesystemCommandPathArguments:
    def test_read_command(self) -> None:
        paths = _filesystem_command_path_arguments("cat", ["file.txt"])
        assert "file.txt" in paths

    def test_write_command(self) -> None:
        paths = _filesystem_command_path_arguments("rm", ["file.txt"])
        assert "file.txt" in paths

    def test_non_filesystem_returns_empty(self) -> None:
        paths = _filesystem_command_path_arguments("echo", ["hello"])
        assert len(paths) == 0

    def test_flags_skipped(self) -> None:
        paths = _filesystem_command_path_arguments("cat", ["-n", "file.txt"])
        assert "file.txt" in paths

    def test_chain_operator_stops(self) -> None:
        paths = _filesystem_command_path_arguments(
            "cat", ["a.txt", "&&", "rm", "b.txt"]
        )
        assert "b.txt" not in paths
