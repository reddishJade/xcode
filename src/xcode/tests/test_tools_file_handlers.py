"""文件处理器纯函数单元测试。"""

from __future__ import annotations

import pytest

from xcode.coding_agent.tools.file_handlers import (
    _parse_offset,
    _parse_limit,
    _prepare_edit_arguments,
    _truncate_line,
    _ensure_write_size,
    _prepare_edits,
    _first_changed_line,
    MAX_WRITE_BYTES,
    MAX_LINE_LENGTH,
    MAX_LINE_SUFFIX,
)


class TestParseOffset:
    def test_default(self) -> None:
        assert _parse_offset({}) == 1

    def test_valid(self) -> None:
        assert _parse_offset({"offset": 10}) == 10

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _parse_offset({"offset": 0})

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            _parse_offset({"offset": "abc"})


class TestParseLimit:
    def test_default(self) -> None:
        assert _parse_limit({}) == 2000

    def test_valid(self) -> None:
        assert _parse_limit({"limit": 50}) == 50

    def test_zero_returns_default(self) -> None:
        assert _parse_limit({"limit": 0}) == 2000

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            _parse_limit({"limit": "abc"})


class TestPrepareEditArguments:
    def test_normal_passthrough(self) -> None:
        data = {"edits": [{"old_text": "a", "new_text": "b"}]}
        result = _prepare_edit_arguments(data)
        assert result["edits"] == [{"old_text": "a", "new_text": "b"}]

    def test_parses_json_edits(self) -> None:
        data = {"edits": '[{"old_text": "a", "new_text": "b"}]'}
        result = _prepare_edit_arguments(data)
        assert result["edits"] == [{"old_text": "a", "new_text": "b"}]

    def test_merges_top_level_old_new_text(self) -> None:
        data = {"old_text": "old", "new_text": "new"}
        result = _prepare_edit_arguments(data)
        assert len(result["edits"]) == 1
        assert result["edits"][0] == {"old_text": "old", "new_text": "new"}
        assert "old_text" not in result
        assert "new_text" not in result

    def test_merges_with_existing_edits(self) -> None:
        data = {
            "edits": [{"old_text": "a", "new_text": "b"}],
            "old_text": "c",
            "new_text": "d",
        }
        result = _prepare_edit_arguments(data)
        assert len(result["edits"]) == 2


class TestTruncateLine:
    def test_short_line(self) -> None:
        assert _truncate_line("hello") == "hello"

    def test_long_line(self) -> None:
        line = "x" * (MAX_LINE_LENGTH + 100)
        result = _truncate_line(line)
        assert len(result) <= MAX_LINE_LENGTH + len(MAX_LINE_SUFFIX)
        assert "truncated" in result


class TestEnsureWriteSize:
    def test_under_limit(self) -> None:
        _ensure_write_size("small")  # should not raise

    def test_over_limit(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            _ensure_write_size("x" * (MAX_WRITE_BYTES + 1))


class TestPrepareEdits:
    def test_from_edits_array(self) -> None:
        data = {"edits": [{"old_text": "a", "new_text": "b"}]}
        edits = _prepare_edits(data)
        assert len(edits) == 1
        assert edits[0].old_text == "a"
        assert edits[0].new_text == "b"

    def test_from_top_level(self) -> None:
        data = {"old_text": "old", "new_text": "new"}
        edits = _prepare_edits(data)
        assert len(edits) == 1
        assert edits[0].old_text == "old"

    def test_empty_old_text_raises(self) -> None:
        data = {"edits": [{"old_text": "", "new_text": "b"}]}
        with pytest.raises(ValueError, match="must not be empty"):
            _prepare_edits(data)

    def test_not_a_dict_in_edits_raises(self) -> None:
        data = {"edits": ["not a dict"]}
        with pytest.raises(ValueError, match="must be an object"):
            _prepare_edits(data)


class TestFirstChangedLine:
    def test_no_change(self) -> None:
        assert _first_changed_line("a\nb", "a\nb") is None

    def test_changed(self) -> None:
        assert _first_changed_line("a\nb", "a\nc") == 2

    def test_extra_line(self) -> None:
        assert _first_changed_line("a", "a\nb") == 2
