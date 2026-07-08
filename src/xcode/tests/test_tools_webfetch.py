"""Web 抓取工具纯函数单元测试。"""

from __future__ import annotations

import pytest
from xcode.coding_agent.tools.webfetch import (
    _valid_url,
    _timeout,
    _format,
    _charset,
    _mime,
    _is_image_mime,
    _is_textual_mime,
    _is_html_mime,
    _plain_text,
    _markdown_text,
    _with_truncation_notice,
    _html_to_markdown,
)


class TestValidUrl:
    def test_valid_https(self) -> None:
        assert _valid_url("https://example.com") == "https://example.com"

    def test_valid_http(self) -> None:
        assert _valid_url("http://example.com") == "http://example.com"

    def test_no_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="HTTP"):
            _valid_url("ftp://example.com")

    def test_no_netloc_raises(self) -> None:
        with pytest.raises(ValueError, match="HTTP"):
            _valid_url("not-a-url")


class TestTimeout:
    def test_default(self) -> None:
        assert _timeout(None) == 30.0

    def test_valid(self) -> None:
        assert _timeout(15) == 15.0

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            _timeout(0)

    def test_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            _timeout(200)

    def test_non_number_raises(self) -> None:
        with pytest.raises(ValueError):
            _timeout("abc")


class TestFormat:
    def test_valid(self) -> None:
        assert _format("markdown") == "markdown"
        assert _format("text") == "text"
        assert _format("html") == "html"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            _format("pdf")


class TestCharset:
    def test_found(self) -> None:
        assert _charset("text/html; charset=utf-8") == "utf-8"

    def test_not_found(self) -> None:
        assert _charset("text/html") == "utf-8"


class TestMime:
    def test_basic(self) -> None:
        assert _mime("text/html; charset=utf-8") == "text/html"

    def test_no_params(self) -> None:
        assert _mime("application/json") == "application/json"


class TestIsImageMime:
    def test_image(self) -> None:
        assert _is_image_mime("image/png")
        assert _is_image_mime("image/jpeg")

    def test_svg_not_image(self) -> None:
        assert not _is_image_mime("image/svg+xml")


class TestIsTextualMime:
    def test_text(self) -> None:
        assert _is_textual_mime("text/plain")

    def test_json(self) -> None:
        assert _is_textual_mime("application/json")

    def test_binary(self) -> None:
        assert not _is_textual_mime("application/octet-stream")

    def test_empty(self) -> None:
        assert _is_textual_mime("")


class TestIsHtmlMime:
    def test_html(self) -> None:
        assert _is_html_mime("text/html")
        assert _is_html_mime("application/xhtml+xml")


class TestPlainText:
    def test_strips_tags(self) -> None:
        result = _plain_text("<p>Hello <b>World</b></p>")
        assert "Hello" in result
        assert "World" in result
        assert "<" not in result


class TestMarkdownText:
    def test_includes_source(self) -> None:
        result = _markdown_text("<p>Content</p>", "https://example.com")
        assert "example.com" in result

    def test_heading_conversion(self) -> None:
        result = _markdown_text("<h1>Title</h1><p>Body</p>", "https://x.com")
        assert "#" in result or "Title" in result


class TestWithTruncationNotice:
    def test_not_truncated(self) -> None:
        assert _with_truncation_notice("hello", False) == "hello"

    def test_truncated_appends_notice(self) -> None:
        result = _with_truncation_notice("hello", True)
        assert "truncated" in result


class TestHtmlToMarkdown:
    def test_removes_script_style(self) -> None:
        html = "<script>alert(1)</script><p>Text</p><style>.c{}</style>"
        result = _html_to_markdown(html)
        assert "alert" not in result
        assert ".c" not in result
        assert "Text" in result

    def test_link_conversion(self) -> None:
        html = '<a href="https://x.com">Click</a>'
        result = _html_to_markdown(html)
        assert "Click" in result
        assert "x.com" in result
