"""Web 搜索工具纯函数单元测试。"""

from __future__ import annotations

import pytest

from xcode.coding_agent.tools.websearch import (
    _limit,
    _livecrawl,
    _parse_mcp_response,
    _search_type,
    _timeout,
)


class TestTimeout:
    def test_default(self) -> None:
        assert _timeout(None) == 30.0

    def test_valid(self) -> None:
        assert _timeout(10) == 10.0


class TestLimit:
    def test_default(self) -> None:
        assert _limit(None) == 8

    def test_valid(self) -> None:
        assert _limit(5) == 5

    def test_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="numResults"):
            _limit(0)

    def test_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="numResults"):
            _limit(50)

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValueError):
            _limit("abc")


class TestLivecrawl:
    def test_default(self) -> None:
        assert _livecrawl(None) == "fallback"

    def test_valid(self) -> None:
        assert _livecrawl("preferred") == "preferred"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="livecrawl"):
            _livecrawl("always")


class TestSearchType:
    def test_default(self) -> None:
        assert _search_type(None) == "auto"

    def test_valid(self) -> None:
        assert _search_type("deep") == "deep"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="type"):
            _search_type("wide")


class TestParseMcpResponse:
    def test_json_result(self) -> None:
        body = '{"result": {"content": [{"type": "text", "text": "Results here"}]}}'
        assert _parse_mcp_response(body) == "Results here"

    def test_sse_format(self) -> None:
        body = (
            'data: {"result": {"content": [{"type": "text", "text": "SSE result"}]}}\n'
        )
        assert _parse_mcp_response(body) == "SSE result"

    def test_empty_body(self) -> None:
        assert _parse_mcp_response("") is None

    def test_no_text_content(self) -> None:
        body = '{"result": {"content": [{"type": "image", "data": "..."}]}}'
        assert _parse_mcp_response(body) is None

    def test_invalid_json(self) -> None:
        assert _parse_mcp_response("not json") is None
