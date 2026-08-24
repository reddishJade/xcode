"""文本编辑纯函数单元测试。"""

from __future__ import annotations

import pytest

from xcode.coding_agent.tools.text_edit import (
    apply_text_replacement,
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)


class TestDetectLineEnding:
    def test_lf(self) -> None:
        assert detect_line_ending("hello\nworld") == "\n"

    def test_crlf(self) -> None:
        assert detect_line_ending("hello\r\nworld") == "\r\n"

    def test_crlf_first(self) -> None:
        text = "hello\r\nworld\nfoo"
        crlf_idx = text.find("\r\n")
        lf_idx = text.find("\n")
        assert crlf_idx < lf_idx
        assert detect_line_ending(text) == "\r\n"

    def test_no_newline(self) -> None:
        assert detect_line_ending("hello") == "\n"


class TestNormalizeToLf:
    def test_crlf_to_lf(self) -> None:
        assert normalize_to_lf("hello\r\nworld") == "hello\nworld"

    def test_cr_to_lf(self) -> None:
        assert normalize_to_lf("hello\rworld") == "hello\nworld"

    def test_already_lf(self) -> None:
        assert normalize_to_lf("hello\nworld") == "hello\nworld"


class TestRestoreLineEndings:
    def test_to_crlf(self) -> None:
        assert restore_line_endings("hello\nworld", "\r\n") == "hello\r\nworld"

    def test_to_lf(self) -> None:
        assert restore_line_endings("hello\nworld", "\n") == "hello\nworld"


class TestStripBom:
    def test_with_bom(self) -> None:
        bom, rest = strip_bom("\ufeffhello")
        assert bom == "\ufeff"
        assert rest == "hello"

    def test_without_bom(self) -> None:
        bom, rest = strip_bom("hello")
        assert bom == ""
        assert rest == "hello"


class TestApplyTextReplacement:
    def test_simple_replacement(self) -> None:
        result = apply_text_replacement("hello world", "hello", "hi")
        assert result == "hi world"

    def test_identical_raises(self) -> None:
        with pytest.raises(ValueError, match="identical"):
            apply_text_replacement("hello", "hello", "hello")

    def test_empty_old_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            apply_text_replacement("hello", "", "new")

    def test_not_found_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not find"):
            apply_text_replacement("hello", "nope", "new")

    def test_multiple_matches_raises(self) -> None:
        with pytest.raises(ValueError, match="multiple matches"):
            apply_text_replacement("a a", "a", "b")

    def test_replace_all(self) -> None:
        result = apply_text_replacement("a a a", "a", "b", replace_all=True)
        assert result == "b b b"
