"""输出截断纯函数单元测试。"""

from __future__ import annotations

from xcode.coding_agent.tools.truncate import (
    truncate_tail,
    truncate_head,
    truncate_line,
    format_size,
    _split_lines,
)


class TestFormatSize:
    def test_bytes(self) -> None:
        assert format_size(500) == "500B"

    def test_kb(self) -> None:
        assert "KB" in format_size(2048)

    def test_mb(self) -> None:
        assert "MB" in format_size(2 * 1024 * 1024)


class TestSplitLines:
    def test_empty(self) -> None:
        assert _split_lines("") == []

    def test_trailing_newline_stripped(self) -> None:
        assert _split_lines("a\nb\n") == ["a", "b"]

    def test_no_trailing_newline(self) -> None:
        assert _split_lines("a\nb") == ["a", "b"]


class TestTruncateHead:
    def test_within_limits(self) -> None:
        r = truncate_head("hello\nworld")
        assert not r.truncated
        assert r.content == "hello\nworld"

    def test_line_limit(self) -> None:
        lines = "\n".join(f"line{i}" for i in range(100))
        r = truncate_head(lines, max_lines=10)
        assert r.truncated
        assert r.truncated_by == "lines"
        assert r.output_lines == 10

    def test_byte_limit(self) -> None:
        lines = "x" * 100_000
        r = truncate_head(lines, max_bytes=1000)
        assert r.truncated
        assert r.truncated_by == "bytes"

    def test_first_line_exceeds_limit(self) -> None:
        r = truncate_head("x" * 100_000, max_bytes=100)
        assert r.content == ""

    def test_empty_content(self) -> None:
        r = truncate_head("")
        assert not r.truncated
        assert r.content == ""


class TestTruncateTail:
    def test_within_limits(self) -> None:
        r = truncate_tail("hello\nworld")
        assert not r.truncated

    def test_line_limit(self) -> None:
        lines = "\n".join(f"line{i}" for i in range(100))
        r = truncate_tail(lines, max_lines=10)
        assert r.truncated
        assert r.truncated_by == "lines"
        assert r.output_lines == 10
        assert r.content.startswith("line90")  # tail keeps last lines

    def test_empty_content(self) -> None:
        r = truncate_tail("")
        assert not r.truncated

    def test_first_line_partial_when_single_line_exceeds(self) -> None:
        content = "x" * 100_000
        r = truncate_tail(content, max_bytes=100)
        assert r.truncated
        assert r.last_line_partial

    def test_total_lines_and_bytes_always_accurate(self) -> None:
        content = "\n".join(f"line{i}" for i in range(50))
        r = truncate_tail(content, max_lines=10)
        assert r.total_lines == 50
        assert r.total_bytes == len(content.encode("utf-8"))


class TestTruncateLine:
    def test_short_line(self) -> None:
        result, truncated = truncate_line("hello")
        assert not truncated
        assert result == "hello"

    def test_long_line(self) -> None:
        result, truncated = truncate_line("x" * 1000, max_chars=100)
        assert truncated
        assert "[truncated]" in result
