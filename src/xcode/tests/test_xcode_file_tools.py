from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import tempfile
from xcode.cli.repl_tools import parse_tool_input
from xcode.coding_agent.tools import build_file_tools
from xcode.coding_agent.tools.file_handlers import LocalFileOperations
import pytest


class RecordingFileOperations(LocalFileOperations):
    def __init__(self) -> None:
        self.reads: list[Path] = []
        self.writes: list[Path] = []
        self.mkdirs: list[Path] = []
        self.line_reads: list[Path] = []
        self.head_reads: list[Path] = []

    def read_bytes(self, path: Path) -> bytes:
        self.reads.append(path)
        return super().read_bytes(path)

    def read_head(self, path: Path, n: int) -> bytes:
        self.head_reads.append(path)
        return super().read_head(path, n)

    def iter_lines(self, path: Path) -> Iterator[str]:
        self.line_reads.append(path)
        return super().iter_lines(path)

    def write_bytes(self, path: Path, data: bytes) -> None:
        self.writes.append(path)
        super().write_bytes(path, data)

    def mkdir(self, path: Path) -> None:
        self.mkdirs.append(path)
        super().mkdir(path)


class XcodeSandboxedFileToolsTests:
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

            assert "1: one\n2: two" in output
            assert "offset=3" in output
            assert "<path>a.txt</path>" in output
            assert "<type>file</type>" in output

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

            assert (
                root / "a.txt" in operations.line_reads
                or root / "a.txt" in operations.reads
            )
            assert root / "b.txt" in operations.writes
            assert root / "a.txt" in operations.writes
            assert root in operations.mkdirs

    def test_apply_patch_adds_updates_deletes_and_moves_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
            (root / "delete.txt").write_text("gone\n", encoding="utf-8")
            (root / "old.txt").write_text("move me\n", encoding="utf-8")
            tools = self._tools(root)

            output = tools["apply_patch"].handler(
                {
                    "patch_text": "\n".join(
                        [
                            "*** Begin Patch",
                            "*** Add File: new.txt",
                            "+created",
                            "*** Update File: a.txt",
                            "@@",
                            " one",
                            "-two",
                            "+three",
                            "*** Delete File: delete.txt",
                            "*** Update File: old.txt",
                            "*** Move to: moved/renamed.txt",
                            "@@",
                            "-move me",
                            "+moved",
                            "*** End Patch",
                        ]
                    )
                }
            )

            assert "A new.txt" in output
            assert "M a.txt" in output
            assert "D delete.txt" in output
            assert "R old.txt -> moved/renamed.txt" in output
            assert (root / "new.txt").read_text(encoding="utf-8") == "created\n"
            assert (root / "a.txt").read_text(encoding="utf-8") == "one\nthree\n"
            assert not (root / "delete.txt").exists()
            assert not (root / "old.txt").exists()
            assert (root / "moved" / "renamed.txt").read_text(
                encoding="utf-8"
            ) == "moved\n"
            metadata = getattr(output, "metadata", {})
            assert "patch" in metadata
            assert len(metadata["files"]) == 4

    def test_apply_patch_rejects_stale_update_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("current\n", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="context not found"):
                tools["apply_patch"].handler(
                    {
                        "patch_text": "\n".join(
                            [
                                "*** Begin Patch",
                                "*** Update File: a.txt",
                                "@@",
                                "-old",
                                "+new",
                                "*** End Patch",
                            ]
                        )
                    }
                )

            assert (root / "a.txt").read_text(encoding="utf-8") == "current\n"

    def test_apply_patch_rejects_move_target_that_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("source\n", encoding="utf-8")
            (root / "target.txt").write_text("target\n", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="move target already exists"):
                tools["apply_patch"].handler(
                    {
                        "patch_text": "\n".join(
                            [
                                "*** Begin Patch",
                                "*** Update File: source.txt",
                                "*** Move to: target.txt",
                                "@@",
                                "-source",
                                "+source",
                                "*** End Patch",
                            ]
                        )
                    }
                )

            assert (root / "source.txt").exists()
            assert (root / "target.txt").read_text(encoding="utf-8") == "target\n"

    def test_read_file_with_offset_and_continuation_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree\nfour", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler(
                {"path": "a.txt", "offset": 2, "limit": 2}
            )

            assert "2: two\n3: three" in output
            assert "offset=4" in output

    def test_read_file_with_offset_without_limit_reads_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "a.txt", "offset": 2})

            assert "2: two\n3: three" in output
            assert "End of file - total 3 lines" in output

    def test_read_file_rejects_invalid_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="limit must be an integer"):
                tools["read_file"].handler({"path": "a.txt", "limit": "bad"})
            with pytest.raises(ValueError, match="limit must be non-negative"):
                tools["read_file"].handler({"path": "a.txt", "limit": -1})

    def test_read_file_rejects_invalid_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="offset must be an integer"):
                tools["read_file"].handler({"path": "a.txt", "offset": "bad"})
            with pytest.raises(ValueError, match="offset must be positive"):
                tools["read_file"].handler({"path": "a.txt", "offset": 0})
            with pytest.raises(ValueError, match="Offset 2 is out of range"):
                tools["read_file"].handler({"path": "a.txt", "offset": 2})

    def test_read_file_accepts_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("hello", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": str(root / "a.txt")})
            assert "1: hello" in output
            assert "End of file" in output

    def test_rejects_parent_and_sensitive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with pytest.raises(ValueError, match="parent-directory"):
                tools["read_file"].handler({"path": "../secret.txt"})
            with pytest.raises(ValueError, match="blocked"):
                tools["read_file"].handler({"path": ".env"})
            with pytest.raises(ValueError, match="blocked"):
                tools["read_file"].handler({"path": "xcode/.local/chroma_db/index"})
            with pytest.raises(ValueError, match="blocked"):
                tools["read_file"].handler({"path": ".local/chroma_db/index"})

    def test_read_file_directory_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file_a.txt").write_text("a", encoding="utf-8")
            (root / "file_b.txt").write_text("b", encoding="utf-8")
            (root / "subdir").mkdir()
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": str(root)})
            assert "<type>directory</type>" in output
            assert "<entries>" in output
            assert "</entries>" in output
            assert "file_a.txt" in output
            assert "file_b.txt" in output
            assert "subdir/" in output
            assert "3 entries" in output

    def test_read_file_directory_listing_with_offset_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                (root / f"file_{i}.txt").write_text(str(i), encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler(
                {"path": str(root), "offset": 2, "limit": 2}
            )
            assert "file_1.txt" in output
            assert "file_2.txt" in output
            assert "file_0.txt" not in output
            assert "file_3.txt" not in output
            assert "Use 'offset' parameter to read beyond entry 4" in output

    def test_read_file_file_not_found_suggests_similar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "readme.txt").write_text("content", encoding="utf-8")
            (root / "reader.py").write_text("code", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError) as exc_info:
                tools["read_file"].handler({"path": "readme.md"})
            msg = str(exc_info.value)
            assert "File not found" in msg
            assert "readme.txt" in msg

    def test_read_file_file_not_found_suggests_similar_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MyFile.txt").write_text("content", encoding="utf-8")
            (root / "myfile.py").write_text("code", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError) as exc_info:
                tools["read_file"].handler({"path": "myfile.md"})
            msg = str(exc_info.value)
            assert "File not found" in msg
            assert "MyFile.txt" in msg or "myfile.py" in msg

    def test_read_file_file_not_found_no_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with pytest.raises(ValueError, match="File not found: nonexistent"):
                tools["read_file"].handler({"path": "nonexistent"})

    def test_read_file_binary_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.zip").write_text("not really zip", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "data.zip"})
            assert "Cannot read binary file" in output
            assert getattr(output, "is_error", False)

    def test_read_file_binary_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = bytes(range(256))
            (root / "mixed.bin").write_bytes(raw)
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "mixed.bin"})
            assert "Cannot read binary file" in output
            assert getattr(output, "is_error", False)

    def test_read_file_structured_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("line1\nline2\nline3", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "a.txt"})
            metadata = getattr(output, "metadata", {})
            display = metadata.get("display", {})
            assert display.get("type") == "file"
            assert display.get("lineStart") == 1
            assert display.get("lineEnd") == 3
            assert display.get("totalLines") == 3
            assert display.get("truncated") is False

    def test_read_file_directory_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "f1.txt").write_text("a", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": str(root)})
            metadata = getattr(output, "metadata", {})
            display = metadata.get("display", {})
            assert display.get("type") == "directory"
            assert display.get("totalEntries") == 1
            assert display.get("truncated") is False

    def test_read_file_with_line_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_line = "x" * 3000
            (root / "long.txt").write_text(long_line, encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "long.txt"})
            assert "... (line truncated to 2000 chars)" in output
            assert len(long_line[:2000]) == 2000

    def test_read_file_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty.txt").write_text("", encoding="utf-8")
            tools = self._tools(root)

            output = tools["read_file"].handler({"path": "empty.txt"})
            assert "End of file - total 0 lines" in output

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
                pytest.skip("symlinks are not available on this platform")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="escapes project root"):
                tools["read_file"].handler({"path": "link.txt"})

    def test_edit_file_requires_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("x\nx\n", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="Found multiple matches"):
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
            assert "replacements=2" in output
            assert "--- a/a.txt" in output
            assert "+++ b/a.txt" in output
            assert path.read_text(encoding="utf-8") == "y\ny\n"

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

            assert "-old" in output
            assert "+new" in output
            assert path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_write_file_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            tools["write_file"].handler({"path": "docs/a.md", "content": "hello"})

            assert (root / "docs" / "a.md").is_file()
            assert (root / "docs" / "a.md").read_text(encoding="utf-8") == "hello"

    def test_write_file_accepts_structured_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            output = tools["write_file"].handler(
                {"path": "docs/a.md", "content": "hello"}
            )

            assert "wrote file: docs/a.md" in output
            assert (root / "docs" / "a.md").read_text(encoding="utf-8") == "hello"

    def test_write_file_requires_path_instead_of_writing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with pytest.raises(ValueError, match="path is required"):
                tools["write_file"].handler({"content": "hello"})

    def test_write_file_requires_content_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with pytest.raises(ValueError, match="content is required"):
                tools["write_file"].handler({"path": "docs/a.md"})

            assert not ((root / "docs" / "a.md").exists())

    def test_write_file_accepts_explicit_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            output = tools["write_file"].handler({"path": "empty.txt", "content": ""})

            assert "wrote file: empty.txt" in output
            assert (root / "empty.txt").read_text(encoding="utf-8") == ""

    def test_write_file_rejects_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            tools = self._tools(root)

            with pytest.raises(ValueError, match="path is a directory: docs"):
                tools["write_file"].handler({"path": "docs", "content": "hello"})

    def test_write_file_rejects_large_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = self._tools(root)

            with pytest.raises(ValueError, match="write content too large"):
                tools["write_file"].handler(
                    {"path": "large.txt", "content": "x" * 1_000_001}
                )

            assert not ((root / "large.txt").exists())

    def test_edit_file_rejects_large_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("small", encoding="utf-8")
            tools = self._tools(root)

            with pytest.raises(ValueError, match="write content too large"):
                tools["edit_file"].handler(
                    {
                        "path": "a.txt",
                        "old_text": "small",
                        "new_text": "x" * 1_000_001,
                    }
                )

            assert path.read_text(encoding="utf-8") == "small"

    def test_file_tools_have_structured_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = self._tools(Path(tmp))

            assert tools["read_file"].schema is not None
            assert tools["write_file"].schema is not None
            assert tools["edit_file"].schema is not None
            assert tools["apply_patch"].schema is not None
            assert "offset" in tools["read_file"].schema["properties"]
            assert tools["write_file"].schema["required"] == ["path", "content"]
            assert tools["edit_file"].schema["required"] == [
                "path",
                "old_text",
                "new_text",
            ]

    def test_file_tools_have_prompt_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = self._tools(Path(tmp))

            write_snippet = tools["write_file"].prompt_snippet
            edit_snippet = tools["edit_file"].prompt_snippet
            assert write_snippet is not None
            assert edit_snippet is not None
            assert "replace entire files" in write_snippet
            assert (
                "Use write_file only for new files"
                in tools["write_file"].prompt_guidelines[0]
            )
            assert "precise file edits" in edit_snippet
            assert any(
                "old_text" in guideline
                for guideline in tools["edit_file"].prompt_guidelines
            )
            assert any(
                "offset and limit" in guideline
                for guideline in tools["read_file"].prompt_guidelines
            )

    def test_file_tool_invalid_json_is_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = self._tools(Path(tmp))["write_file"]
            with pytest.raises(ValueError, match="invalid JSON input"):
                parse_tool_input(tool, '{"path": "a.txt",')

    def test_file_tool_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = self._tools(Path(tmp))["read_file"]
            with pytest.raises(ValueError, match="JSON input must be an object"):
                parse_tool_input(tool, '["a.txt"]')

    def test_edit_file_external_modification_uses_old_text_matching(self) -> None:
        """edit_file 不依赖哈希校验，外部修改后 old_text 不匹配时返回错误。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("original content", encoding="utf-8")
            tools = self._tools(root)

            path.write_text("modified externally", encoding="utf-8")
            with pytest.raises(ValueError, match="Could not find old_string"):
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

            assert "replacements=1" in output
            assert path.read_text(encoding="utf-8") == "hi world"

    def test_edit_file_result_includes_patch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("one\ntwo\n", encoding="utf-8")
            tools = self._tools(root)

            output = tools["edit_file"].handler(
                {"path": "a.txt", "old_text": "two", "new_text": "three"}
            )

            assert "-two" in output
            assert "+three" in output
            assert getattr(output, "metadata", {}).get("first_changed_line") == 2
            assert path.read_text(encoding="utf-8") == "one\nthree\n"

    def test_edit_file_requires_new_text_key_but_allows_empty_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "a.txt"
            path.write_text("delete me", encoding="utf-8")
            tools = self._tools(root)

            tools["read_file"].handler({"path": "a.txt"})
            with pytest.raises(ValueError, match="new_text is required"):
                tools["edit_file"].handler({"path": "a.txt", "old_text": "delete me"})
            deleted = tools["edit_file"].handler(
                {"path": "a.txt", "old_text": "delete me", "new_text": ""}
            )

            assert "replacements=1" in deleted
            assert path.read_text(encoding="utf-8") == ""


if __name__ == "__main__":
    pytest.main()
