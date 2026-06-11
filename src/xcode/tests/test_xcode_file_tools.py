from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xcode.cli.repl_tools import parse_tool_input
from xcode.harness.observability import HITLResult
from xcode.harness.skills import run_tool_result
from xcode.tests.fixtures import run_tool
from xcode.coding_agent.tools import build_file_tools
from xcode.coding_agent.tools.file import LocalFileOperations


class RecordingFileOperations(LocalFileOperations):
    def __init__(self) -> None:
        self.reads: list[Path] = []
        self.writes: list[Path] = []
        self.mkdirs: list[Path] = []

    def read_bytes(self, path: Path) -> bytes:
        self.reads.append(path)
        return super().read_bytes(path)

    def write_bytes(self, path: Path, data: bytes) -> None:
        self.writes.append(path)
        super().write_bytes(path, data)

    def mkdir(self, path: Path) -> None:
        self.mkdirs.append(path)
        super().mkdir(path)


class XcodeSandboxedFileToolsTests(unittest.TestCase):
    def _tools(self, root: Path):
        return {tool.name: tool for tool in build_file_tools(root)}

    def _tools_with_operations(self, root: Path, operations: RecordingFileOperations):
        return {
            tool.name: tool for tool in build_file_tools(root, operations=operations)
        }

    def test_read_file_with_limit_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "a.txt", "limit": 2})

            self.assertIn("one\ntwo", output)
            self.assertIn('"offset": 3', output)

    def test_file_tools_use_injected_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("old", encoding="utf-8")
            operations = RecordingFileOperations()
            tools = self._tools_with_operations(root, operations)

            tools["read_file"].handler({"path": "a.txt"})
            tools["write_file"].handler({"path": "b.txt", "content": "new"})
            tools["edit_file"].handler(
                {"path": "a.txt", "old_text": "old", "new_text": "updated"}
            )

            self.assertIn(root / "a.txt", operations.reads)
            self.assertIn(root / "b.txt", operations.writes)
            self.assertIn(root / "a.txt", operations.writes)
            self.assertIn(root, operations.mkdirs)

    def test_read_file_with_offset_and_continuation_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree\nfour", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler(
                {"path": "a.txt", "offset": 2, "limit": 2}
            )

            self.assertIn("two\nthree", output)
            self.assertIn('"offset": 4', output)
            self.assertIn('"limit": 2', output)

    def test_read_file_with_offset_without_limit_reads_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "a.txt", "offset": 2})

            self.assertEqual(output, "two\nthree")

    def test_read_file_rejects_invalid_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one", encoding="utf-8")
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "limit must be an integer"):
                tools["read_file"].handler({"path": "a.txt", "limit": "bad"})
            with self.assertRaisesRegex(ValueError, "limit must be non-negative"):
                tools["read_file"].handler({"path": "a.txt", "limit": -1})

    def test_read_file_rejects_invalid_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one", encoding="utf-8")
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "offset must be an integer"):
                tools["read_file"].handler({"path": "a.txt", "offset": "bad"})
            with self.assertRaisesRegex(ValueError, "offset must be positive"):
                tools["read_file"].handler({"path": "a.txt", "offset": 0})
            with self.assertRaisesRegex(ValueError, "beyond end of file"):
                tools["read_file"].handler({"path": "a.txt", "offset": 2})

    def test_rejects_absolute_parent_and_sensitive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "absolute"):
                tools["read_file"].handler({"path": str(root / "a.txt")})
            with self.assertRaisesRegex(ValueError, "parent-directory"):
                tools["read_file"].handler({"path": "../secret.txt"})
            with self.assertRaisesRegex(ValueError, "blocked"):
                tools["read_file"].handler({"path": ".env"})
            with self.assertRaisesRegex(ValueError, "blocked"):
                tools["read_file"].handler({"path": "xcode/.local/chroma_db/index"})
            with self.assertRaisesRegex(ValueError, "blocked"):
                tools["read_file"].handler({"path": ".local/chroma_db/index"})

    def test_rejects_symlink_escape(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            root = Path(tmp)
            target = Path(outside) / "secret.txt"
            target.write_text("secret", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are not available on this platform")
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "escapes project root"):
                tools["read_file"].handler({"path": "link.txt"})

    def test_edit_file_requires_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("x\nx\n", encoding="utf-8")
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "Found 2 occurrences"):
                tools["edit_file"].handler(
                    {"path": "a.txt", "old_text": "x", "new_text": "y"}
                )

            output = tools["edit_file"].handler(
                {
                    "path": "a.txt",
                    "old_text": "x",
                    "new_text": "y",
                    "replace_all": True,
                }
            )
            self.assertIn("replacements=2", output)
            self.assertIn("--- a/a.txt", output)
            self.assertIn("+++ b/a.txt", output)
            self.assertEqual(path.read_text(encoding="utf-8"), "y\ny\n")

    def test_edit_file_preserves_utf8_sig_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("old", encoding="utf-8-sig")
            tools = self._tools(root)

            tools["read_file"].handler({"path": "a.txt"})
            output = tools["edit_file"].handler(
                {"path": "a.txt", "old_text": "old", "new_text": "new"}
            )

            self.assertIn("-old", output)
            self.assertIn("+new", output)
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_write_file_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            tools["write_file"].handler({"path": "docs/a.md", "content": "hello"})

            self.assertTrue((root / "docs" / "a.md").is_file())
            self.assertEqual(
                (root / "docs" / "a.md").read_text(encoding="utf-8"),
                "hello",
            )

    def test_write_file_accepts_structured_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            output = run_tool(
                tools,
                "write_file",
                {"path": "docs/a.md", "content": "hello"},
                lambda _tool, _input: HITLResult("allow", "once"),
            )

            self.assertIn("wrote file: docs/a.md", output)
            self.assertEqual(
                (root / "docs" / "a.md").read_text(encoding="utf-8"), "hello"
            )

    def test_write_file_requires_path_instead_of_writing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "path is required"):
                tools["write_file"].handler({"content": "hello"})

    def test_write_file_requires_content_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "content is required"):
                tools["write_file"].handler({"path": "docs/a.md"})

            self.assertFalse((root / "docs" / "a.md").exists())

    def test_write_file_accepts_explicit_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            output = tools["write_file"].handler({"path": "empty.txt", "content": ""})

            self.assertIn("wrote file: empty.txt", output)
            self.assertEqual((root / "empty.txt").read_text(encoding="utf-8"), "")

    def test_write_file_rejects_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "path is a directory: docs"):
                tools["write_file"].handler({"path": "docs", "content": "hello"})

    def test_write_file_rejects_large_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "write content too large"):
                tools["write_file"].handler(
                    {"path": "large.txt", "content": "x" * 1_000_001}
                )

            self.assertFalse((root / "large.txt").exists())

    def test_edit_file_rejects_large_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("small", encoding="utf-8")
            tools = self._tools(root)

            with self.assertRaisesRegex(ValueError, "write content too large"):
                tools["edit_file"].handler(
                    {
                        "path": "a.txt",
                        "old_text": "small",
                        "new_text": "x" * 1_000_001,
                    }
                )

            self.assertEqual(path.read_text(encoding="utf-8"), "small")

    def test_file_tools_have_structured_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = self._tools(Path(tmp))

            assert tools["read_file"].schema is not None
            assert tools["write_file"].schema is not None
            assert tools["edit_file"].schema is not None
            self.assertIn("offset", tools["read_file"].schema["properties"])
            self.assertEqual(
                tools["write_file"].schema["required"], ["path", "content"]
            )
            self.assertEqual(tools["edit_file"].schema["required"], ["path"])

    def test_file_tools_have_prompt_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = self._tools(Path(tmp))

            write_snippet = tools["write_file"].prompt_snippet
            edit_snippet = tools["edit_file"].prompt_snippet
            assert write_snippet is not None
            assert edit_snippet is not None
            self.assertIn("replace entire files", write_snippet)
            self.assertIn(
                "Use write_file only for new files",
                tools["write_file"].prompt_guidelines[0],
            )
            self.assertIn("precise file edits", edit_snippet)
            self.assertTrue(
                any(
                    "multiple entries in edits" in guideline
                    for guideline in tools["edit_file"].prompt_guidelines
                )
            )
            self.assertTrue(
                any(
                    "offset and limit" in guideline
                    for guideline in tools["read_file"].prompt_guidelines
                )
            )

    def test_file_tool_invalid_json_is_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = self._tools(Path(tmp))["write_file"]
            with self.assertRaisesRegex(ValueError, "invalid JSON input"):
                parse_tool_input(tool, '{"path": "a.txt",')

    def test_file_tool_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = self._tools(Path(tmp))["read_file"]
            with self.assertRaisesRegex(ValueError, "JSON input must be an object"):
                parse_tool_input(tool, '["a.txt"]')

    def test_edit_file_external_modification_uses_old_text_matching(self) -> None:
        """edit_file 不依赖哈希校验，外部修改后 old_text 不匹配时返回错误。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("original content", encoding="utf-8")
            tools = self._tools(root)

            path.write_text("modified externally", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Could not find the exact text"):
                tools["edit_file"].handler(
                    {
                        "path": "a.txt",
                        "old_text": "original content",
                        "new_text": "new content",
                    }
                )

    def test_edit_file_works_when_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("hello world", encoding="utf-8")
            tools = self._tools(root)

            tools["read_file"].handler({"path": "a.txt"})
            output = tools["edit_file"].handler(
                {"path": "a.txt", "old_text": "hello", "new_text": "hi"}
            )

            self.assertIn("replacements=1", output)
            self.assertEqual(path.read_text(encoding="utf-8"), "hi world")

    def test_edit_file_result_includes_patch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("one\ntwo\n", encoding="utf-8")
            tools = self._tools(root)

            result = run_tool_result(
                tools,
                "edit_file",
                {"path": "a.txt", "old_text": "two", "new_text": "three"},
                lambda _tool, _input: HITLResult("allow", "once"),
            )

            metadata = result.metadata or {}
            patch = metadata.get("patch")
            assert isinstance(patch, str)
            self.assertEqual(result.status, "ok")
            self.assertIn("-two", patch)
            self.assertIn("+three", patch)
            self.assertEqual(metadata.get("first_changed_line"), 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\nthree\n")

    def test_edit_file_requires_new_text_key_but_allows_empty_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("delete me", encoding="utf-8")
            tools = self._tools(root)

            tools["read_file"].handler({"path": "a.txt"})
            with self.assertRaisesRegex(ValueError, "new_text is required"):
                tools["edit_file"].handler({"path": "a.txt", "old_text": "delete me"})
            deleted = tools["edit_file"].handler(
                {"path": "a.txt", "old_text": "delete me", "new_text": ""}
            )

            self.assertIn("replacements=1", deleted)
            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
