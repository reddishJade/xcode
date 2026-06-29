"""代码搜索工具契约测试。"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import Mock, patch

from xcode.coding_agent.tools import build_glob_tools, build_grep_tool
from xcode.harness.skills import ToolSpec
import pytest


class XcodeCodeToolsTests:
    """验证代码搜索工具的路径、ignore、rg 优先回退契约。"""

    def test_glob_files_finds_project_paths(self) -> None:
        """glob_files 返回匹配的项目相对路径。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "a.py").write_text("", encoding="utf-8")
            (root / "pkg" / "b.txt").write_text("", encoding="utf-8")
            tools = _tools(root)
            output = tools["glob_files"].handler(
                {"path": "pkg", "pattern": "*.py", "max_results": 10}
            )

            assert output == "pkg/a.py"

    def test_glob_files_rejects_blocked_paths(self) -> None:
        """glob_files 拒绝 blocked path。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("TOKEN=x", encoding="utf-8")
            tools = _tools(root)

            with pytest.raises(ValueError):
                tools["glob_files"].handler({"path": ".env", "pattern": "*"})

    def test_glob_files_recursive_finds_subdir_files(self) -> None:
        """递归 glob 返回嵌套目录文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "a.md").write_text("hello", encoding="utf-8")
            (root / "docs" / "nested").mkdir()
            (root / "docs" / "nested" / "b.md").write_text("world", encoding="utf-8")
            tools = _tools(root)
            output = tools["glob_files"].handler(
                {"path": ".", "pattern": "**/*.md", "max_results": 10}
            )

            assert "docs/a.md" in output
            assert "docs/nested/b.md" in output

    def test_glob_files_empty_directory_returns_no_files(self) -> None:
        """空目录返回 No files found。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = _tools(Path(temp_dir))
            output = tools["glob_files"].handler({"pattern": "*.py"})

            assert "No files found" in output

    def test_grep_search_finds_project_text(self) -> None:
        """grep_search 返回项目文本匹配。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "a.py").write_text(
                "class Agent:\n    pass\n", encoding="utf-8"
            )
            tools = _tools(root)
            output = tools["grep_search"].handler(
                {"pattern": "Agent", "path": ".", "glob": "*.py"}
            )

            assert "a.py" in output
            assert "Agent" in output

    def test_grep_search_fallback_works_without_rg(self) -> None:
        """缺少 rg 时自动使用 Python fallback，不抛错。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("needle\n", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["grep_search"].handler({"pattern": "needle"})

            assert "a.txt" in output
            assert "needle" in output

    def test_glob_files_fallback_works_without_rg(self) -> None:
        """缺少 rg 时 glob_files 自动使用 Python fallback。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.py").write_text("", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["glob_files"].handler({"pattern": "*.py"})

            assert "a.py" in output

    def test_find_files_fallback_works_without_rg(self) -> None:
        """缺少 rg 时 find_files 自动使用 Python fallback。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.py").write_text("", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["find_files"].handler({"pattern": "*.py"})

            assert "a.py" in output

    def test_grep_search_rejects_blocked_paths(self) -> None:
        """grep_search 拒绝 blocked path。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("TOKEN=x", encoding="utf-8")
            tools = _tools(root)

            with pytest.raises(ValueError):
                tools["grep_search"].handler({"pattern": "TOKEN", "path": ".env"})

    def test_search_tools_exclude_hidden_and_blocked(self) -> None:
        """搜索工具排除 hidden 和 blocked 文件（无 rg fallback）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "visible.py").write_text("needle", encoding="utf-8")
            (root / ".hidden.py").write_text("needle", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                glob_output = tools["glob_files"].handler(
                    {"pattern": "**/*.py", "max_results": 20}
                )
                find_output = tools["find_files"].handler(
                    {"pattern": "*.py", "max_results": 20}
                )
                grep_output = tools["grep_search"].handler(
                    {"pattern": "needle", "glob": "*.py"}
                )

            for output in (glob_output, find_output):
                assert "visible.py" in output
                assert ".hidden.py" not in output

            assert "visible.py" in grep_output
            assert ".hidden.py" not in grep_output

    def test_search_tools_exclude_gitignored_without_rg(self) -> None:
        """Python fallback 排除 .gitignore 文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
            (root / "visible.py").write_text("needle", encoding="utf-8")
            (root / "ignored.py").write_text("needle", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                glob_output = tools["glob_files"].handler(
                    {"pattern": "**/*.py", "max_results": 20}
                )
                find_output = tools["find_files"].handler(
                    {"pattern": "*.py", "max_results": 20}
                )
                grep_output = tools["grep_search"].handler(
                    {"pattern": "needle", "glob": "*.py"}
                )

            for output in (glob_output, find_output, grep_output):
                assert "visible.py" in output
                assert "ignored.py" not in output

    def test_python_fallback_applies_nested_gitignore_scope(self) -> None:
        """嵌套 .gitignore 仅影响其所在目录树。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "other").mkdir()
            (root / "pkg" / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
            (root / "pkg" / "ignored.tmp").write_text("", encoding="utf-8")
            (root / "other" / "visible.tmp").write_text("", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["find_files"].handler(
                    {"pattern": "*.tmp", "max_results": 10}
                )

            assert output == "other/visible.tmp"

    def test_search_tools_validate_numeric_bounds(self) -> None:
        """搜索工具拒绝非整数和越界数值参数。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = _tools(Path(temp_dir))

            with pytest.raises(ValueError, match="max_results must be an integer"):
                tools["glob_files"].handler({"max_results": "10"})
            with pytest.raises(ValueError, match="max_results must be at least 1"):
                tools["find_files"].handler({"pattern": "*", "max_results": 0})
            with pytest.raises(ValueError, match="context must be at least 0"):
                tools["grep_search"].handler({"pattern": "x", "context": -1})

    def test_grep_fallback_reports_invalid_regex(self) -> None:
        """Python fallback 中非法正则返回明确诊断。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("text", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["grep_search"].handler({"pattern": "["})

            assert "Invalid regex pattern" in output
            assert output != "No matches found."

    def test_ripgrep_discovery_error_is_not_reported_as_no_files(self) -> None:
        """ripgrep 文件枚举失败时保留 stderr 诊断。"""
        completed = subprocess.CompletedProcess(
            args=["rg"],
            returncode=2,
            stdout="",
            stderr="invalid glob",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch(
                    "xcode.coding_agent.tools._search_utils.get_rg_path",
                    return_value="rg",
                ),
                patch(
                    "xcode.coding_agent.tools.glob_search.subprocess.run",
                    return_value=completed,
                ),
            ):
                tools = _tools(root)
                with pytest.raises(
                    ValueError,
                    match="ripgrep file discovery failed: invalid glob",
                ):
                    tools["glob_files"].handler({"pattern": "*.py"})

    def test_ripgrep_search_error_is_not_reported_as_no_matches(self) -> None:
        """ripgrep 内容搜索失败时保留 stderr 诊断。"""
        process = Mock()
        process.stdout = StringIO("")
        process.stderr = StringIO("regex parse error")
        process.returncode = 2
        process.wait.return_value = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = _tools(Path(temp_dir))
            with (
                patch(
                    "xcode.coding_agent.tools._search_utils.get_rg_path",
                    return_value="rg",
                ),
                patch(
                    "xcode.coding_agent.tools.grep_search.subprocess.Popen",
                    return_value=process,
                ),
            ):
                with pytest.raises(
                    ValueError,
                    match="ripgrep failed: regex parse error",
                ):
                    tools["grep_search"].handler({"pattern": "["})

    def test_grep_search_returns_structured_metadata(self) -> None:
        """grep_search 返回 matches 和 truncated 元数据。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("hello world\n", encoding="utf-8")
            tools = _tools(root)
            output = tools["grep_search"].handler({"pattern": "hello"})

            metadata = getattr(output, "metadata", {})
            assert metadata.get("matches") == 1
            assert metadata.get("truncated") is False

    def test_glob_files_returns_structured_metadata(self) -> None:
        """glob_files 返回 count 和 truncated 元数据。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.py").write_text("", encoding="utf-8")
            (root / "b.py").write_text("", encoding="utf-8")
            tools = _tools(root)
            output = tools["glob_files"].handler({"pattern": "*.py", "max_results": 10})

            metadata = getattr(output, "metadata", {})
            assert metadata.get("count") == 2
            assert metadata.get("truncated") is False

    def test_list_dir_returns_structured_metadata(self) -> None:
        """list_dir 返回 count 和 truncated 元数据。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("", encoding="utf-8")
            (root / "b.txt").write_text("", encoding="utf-8")
            tools = _tools(root)

            output = tools["list_dir"].handler({"path": str(root)})

            metadata = getattr(output, "metadata", {})
            assert metadata.get("count") == 2
            assert metadata.get("truncated") is False

    def test_list_dir_empty_directory(self) -> None:
        """空目录返回 (empty directory)。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = _tools(Path(temp_dir))

            output = tools["list_dir"].handler({"path": str(Path(temp_dir))})

            assert "(empty directory)" in output
            metadata = getattr(output, "metadata", {})
            assert metadata.get("count") == 0

    def test_grep_supports_ignore_case_and_literal(self) -> None:
        """grep_search 支持 ignore_case 和 literal 参数（fallback）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("HELLO World\n", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                case_output = tools["grep_search"].handler(
                    {"pattern": "hello", "ignore_case": True}
                )
                literal_output = tools["grep_search"].handler(
                    {"pattern": "HELLO", "literal": True}
                )

            assert "HELLO" in case_output
            assert "HELLO" in literal_output

    def test_glob_files_sorts_by_mtime_then_path_fallback(self) -> None:
        """Python fallback 按 mtime 降序、同名按路径排序。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.py"
            second = root / "b.py"
            newest = root / "c.py"
            for path in (first, second, newest):
                path.write_text("", encoding="utf-8")
            os.utime(first, ns=(1_000_000_000, 1_000_000_000))
            os.utime(second, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newest, ns=(2_000_000_000, 2_000_000_000))
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["glob_files"].handler(
                    {"pattern": "*.py", "max_results": 10}
                )

            assert output.splitlines() == ["c.py", "a.py", "b.py"]

    def test_python_fallback_works_without_git_repository(self) -> None:
        """无 .git 目录时 Python fallback 仍应用 .gitignore。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
            (root / "visible.txt").write_text("visible", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")
            tools = _tools(root)

            with patch(
                "xcode.coding_agent.tools._search_utils.get_rg_path",
                return_value=None,
            ):
                output = tools["find_files"].handler(
                    {"pattern": "*.txt", "max_results": 10}
                )

            assert output == "visible.txt"


def _tools(root: Path) -> dict[str, ToolSpec]:
    """按名称索引项目代码搜索工具。"""
    tools: dict[str, ToolSpec] = {}
    for tool in build_glob_tools(root):
        tools[tool.name] = tool
    tool = build_grep_tool(root)
    tools[tool.name] = tool
    return tools


if __name__ == "__main__":
    pytest.main()
