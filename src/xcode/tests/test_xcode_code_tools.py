"""代码搜索工具契约测试。"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import Mock, patch

import xcode.coding_agent.tools.code_search as code_search
from xcode.coding_agent.tools import build_code_tools
from xcode.harness.skills import ToolSpec
import pytest
class XcodeCodeToolsTests:
    """验证代码搜索工具的路径、ignore、排序和诊断契约。"""

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
        """ripgrep 对空目录返回 1 时仍表示正常无结果。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = _tools(Path(temp_dir))

            output = tools["glob_files"].handler({"pattern": "*.py"})

        assert output == "No files found."

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

    def test_grep_search_without_rg_warns_once_then_falls_back(self) -> None:
        """缺少 ripgrep 时只提示一次并使用 Python fallback。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("needle\n", encoding="utf-8")
            tools = _tools(root)
            code_search._RG_MISSING_HINT_EMITTED = False

            with patch(
                "xcode.coding_agent.tools.code_search.ensure_tool", return_value=None
            ):
                first = tools["grep_search"].handler({"pattern": "needle"})
                second = tools["grep_search"].handler({"pattern": "needle"})

            assert "ripgrep not found" in first
            assert "a.txt" in first
            assert "ripgrep not found" not in second

    def test_grep_search_rejects_blocked_paths(self) -> None:
        """grep_search 拒绝 blocked path。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("TOKEN=x", encoding="utf-8")
            tools = _tools(root)

            with pytest.raises(ValueError):
                tools["grep_search"].handler({"pattern": "TOKEN", "path": ".env"})

    def test_search_tools_exclude_ignored_and_hidden_files_without_rg(self) -> None:
        """Python fallback 与外部工具路径遵循相同 ignore 和 hidden 规则。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".gitignore").write_text("ignored.py\nbuild/\n", encoding="utf-8")
            (root / "visible.py").write_text("needle", encoding="utf-8")
            (root / "ignored.py").write_text("needle", encoding="utf-8")
            (root / ".hidden.py").write_text("needle", encoding="utf-8")
            (root / "build").mkdir()
            (root / "build" / "generated.py").write_text("needle", encoding="utf-8")
            tools = _tools(root)
            code_search._RG_MISSING_HINT_EMITTED = True

            with patch(
                "xcode.coding_agent.tools.code_search.ensure_tool",
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
            assert ".hidden.py" not in output
            assert "generated.py" not in output

    def test_glob_files_sorts_by_mtime_then_path(self) -> None:
        """glob 结果按修改时间降序，同时间按路径稳定排序。"""
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
                "xcode.coding_agent.tools.code_search.ensure_tool",
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

            primary_output = tools["find_files"].handler(
                {"pattern": "*.txt", "max_results": 10}
            )
            primary_grep = tools["grep_search"].handler(
                {"pattern": "visible|ignored", "glob": "*.txt"}
            )
            with patch(
                "xcode.coding_agent.tools.code_search.ensure_tool",
                return_value=None,
            ):
                fallback_output = tools["find_files"].handler(
                    {"pattern": "*.txt", "max_results": 10}
                )

        assert primary_output == "visible.txt"
        assert fallback_output == "visible.txt"
        assert "visible.txt" in primary_grep
        assert "ignored.txt" not in primary_grep

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
                "xcode.coding_agent.tools.code_search.ensure_tool",
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
        """无 ripgrep 时非法正则返回明确诊断。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("text", encoding="utf-8")
            tools = _tools(root)
            code_search._RG_MISSING_HINT_EMITTED = True

            with patch(
                "xcode.coding_agent.tools.code_search.ensure_tool",
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
            tools = _tools(root)
            with (
                patch(
                    "xcode.coding_agent.tools.code_search.ensure_tool",
                    return_value="rg",
                ),
                patch(
                    "xcode.coding_agent.tools.code_search.subprocess.run",
                    return_value=completed,
                ),
            ):
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
                    "xcode.coding_agent.tools.code_search.ensure_tool",
                    return_value="rg",
                ),
                patch(
                    "xcode.coding_agent.tools.code_search.subprocess.Popen",
                    return_value=process,
                ),
            ):
                with pytest.raises(
                    ValueError,
                    match="ripgrep failed: regex parse error",
                ):
                    tools["grep_search"].handler({"pattern": "["})

def _tools(root: Path) -> dict[str, ToolSpec]:
    """按名称索引项目代码搜索工具。"""
    return {tool.name: tool for tool in build_code_tools(root)}

if __name__ == "__main__":
    pytest.main()
