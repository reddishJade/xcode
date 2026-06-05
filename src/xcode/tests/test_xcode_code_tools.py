from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import xcode.coding_agent.tools.code_search as code_search
from xcode.coding_agent.tools import build_code_tools


class XcodeCodeToolsTests(unittest.TestCase):
    def test_glob_files_finds_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "a.py").write_text("", encoding="utf-8")
            (root / "pkg" / "b.txt").write_text("", encoding="utf-8")
            tools = _tools(root)

            output = tools["glob_files"].handler(
                {"path": "pkg", "pattern": "*.py", "max_results": 10}
            )

            self.assertEqual(output, "pkg/a.py")

    def test_glob_files_rejects_blocked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("TOKEN=x", encoding="utf-8")
            tools = _tools(root)

            with self.assertRaises(ValueError):
                tools["glob_files"].handler({"path": ".env", "pattern": "*"})

    def test_glob_files_recursive_finds_subdir_files(self) -> None:
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

            self.assertIn("docs/a.md", output)
            self.assertIn("docs/nested/b.md", output)

    def test_grep_search_finds_project_text(self) -> None:
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

            self.assertIn("a.py", output)
            self.assertIn("Agent", output)

    def test_grep_search_without_rg_warns_once_then_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("needle\n", encoding="utf-8")
            tools = _tools(root)
            code_search._RG_MISSING_HINT_EMITTED = False

            with patch(
                "xcode.coding_agent.tools.code_search.shutil.which", return_value=None
            ):
                first = tools["grep_search"].handler({"pattern": "needle"})
                second = tools["grep_search"].handler({"pattern": "needle"})

            self.assertIn("ripgrep not found", first)
            self.assertIn("a.txt", first)
            self.assertNotIn("ripgrep not found", second)

    def test_grep_search_rejects_blocked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("TOKEN=x", encoding="utf-8")
            tools = _tools(root)

            with self.assertRaises(ValueError):
                tools["grep_search"].handler({"pattern": "TOKEN", "path": ".env"})


def _tools(root: Path):
    return {tool.name: tool for tool in build_code_tools(root)}


if __name__ == "__main__":
    unittest.main()
