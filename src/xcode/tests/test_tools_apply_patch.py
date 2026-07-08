"""apply_patch 工具解析与纯函数单元测试。"""

from __future__ import annotations

import pytest
from xcode.coding_agent.tools.apply_patch import (
    parse_patch,
    _header_path,
    _patch_text,
    _find_sequence,
    _find_anchor,
    _change_counts,
    extract_patch_paths,
)


class TestParsePatch:
    def test_add_file(self) -> None:
        patch = """*** Begin Patch
*** Add File: src/new.py
+print('hello')
*** End Patch"""
        hunks = parse_patch(patch)
        assert len(hunks) == 1
        assert hunks[0].kind == "add"
        assert hunks[0].path == "src/new.py"
        assert hunks[0].add_lines == ("print('hello')",)

    def test_delete_file(self) -> None:
        patch = """*** Begin Patch
*** Delete File: src/old.py
*** End Patch"""
        hunks = parse_patch(patch)
        assert len(hunks) == 1
        assert hunks[0].kind == "delete"
        assert hunks[0].path == "src/old.py"

    def test_update_file(self) -> None:
        patch = """*** Begin Patch
*** Update File: src/main.py
@@
-old line
+new line
*** End Patch"""
        hunks = parse_patch(patch)
        assert len(hunks) == 1
        assert hunks[0].kind == "update"

    def test_move_file(self) -> None:
        patch = """*** Begin Patch
*** Update File: src/old.py
*** Move to: src/new.py
*** End Patch"""
        hunks = parse_patch(patch)
        assert hunks[0].kind == "move"
        assert hunks[0].move_path == "src/new.py"

    def test_missing_begin_raises(self) -> None:
        with pytest.raises(ValueError, match="Begin Patch"):
            parse_patch("*** End Patch")

    def test_missing_end_raises(self) -> None:
        with pytest.raises(ValueError, match="End Patch"):
            parse_patch("*** Begin Patch")

    def test_empty_patch_raises(self) -> None:
        with pytest.raises(ValueError, match="rejected"):
            parse_patch("*** Begin Patch\n*** End Patch")


class TestHeaderPath:
    def test_extracts_path(self) -> None:
        assert _header_path("*** Update File: src/main.py", "*** Update File: ") == "src/main.py"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _header_path("*** Update File: ", "*** Update File: ")


class TestPatchText:
    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="patch_text"):
            _patch_text({})

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="patch_text"):
            _patch_text({"patch_text": ""})


class TestFindSequence:
    def test_found(self) -> None:
        lines = ["a", "b", "c", "d"]
        assert _find_sequence(lines, ("b", "c"), 0) == 1

    def test_not_found(self) -> None:
        assert _find_sequence(["a", "b"], ("c",), 0) is None

    def test_empty_needle_returns_start(self) -> None:
        assert _find_sequence(["a"], (), 2) == 2

    def test_beyond_bounds_returns_none(self) -> None:
        assert _find_sequence(["a"], ("a",), 5) is None


class TestFindAnchor:
    def test_after_cursor(self) -> None:
        lines = ["a", "b", "c"]
        assert _find_anchor(lines, "b", 0) == 2

    def test_fallback_from_start(self) -> None:
        lines = ["a", "b", "c"]
        assert _find_anchor(lines, "b", 5) == 2

    def test_empty_anchor_returns_cursor(self) -> None:
        assert _find_anchor(["a"], "", 1) == 1


class TestChangeCounts:
    def test_insertion(self) -> None:
        a, d = _change_counts("a\nb", "a\nb\nc")
        assert a == 1
        assert d == 0

    def test_deletion(self) -> None:
        a, d = _change_counts("a\nb\nc", "a\nb")
        assert a == 0
        assert d == 1

    def test_replacement(self) -> None:
        a, d = _change_counts("old\n", "new\n")
        assert a >= 1
        assert d >= 1


class TestExtractPatchPaths:
    def test_from_patch_text(self) -> None:
        data = {
            "patch_text": "*** Begin Patch\n*** Update File: src/a.py\n@@\n-old\n+new\n*** End Patch",
        }
        paths = extract_patch_paths(data)
        assert "src/a.py" in paths

    def test_from_paths_field(self) -> None:
        data = {"paths": ["src/a.py", "src/b.py"]}
        paths = extract_patch_paths(data)
        assert len(paths) == 2

    def test_deduplicates(self) -> None:
        data = {
            "paths": ["src/a.py"],
            "patch_text": "*** Begin Patch\n*** Update File: src/a.py\n*** End Patch",
        }
        paths = extract_patch_paths(data)
        assert len(paths) == 1

    def test_non_dict_returns_empty(self) -> None:
        assert extract_patch_paths("not a dict") == ()
