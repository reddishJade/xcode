"""webfetch / websearch 工具单元测试。"""

from __future__ import annotations

import gzip
import io
import json
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from xcode.coding_agent.tools import build_webfetch_tool, build_websearch_tool


@pytest.fixture
def web_tools() -> dict[str, Any]:
    return {
        "webfetch": build_webfetch_tool(),
        "websearch": build_websearch_tool(),
    }


def _make_response(body: bytes, content_type: str, encoding: str | None = None) -> Any:
    response = MagicMock()
    response.headers = {"content-type": content_type}
    if encoding:
        response.headers["content-encoding"] = encoding
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


class TestWebFetch:
    def test_gzip_html_is_decompressed(self, web_tools: dict[str, Any]) -> None:
        html = "<html><body><h1>Hello</h1><p>world</p></body></html>"
        compressed = gzip.compress(html.encode("utf-8"))
        response = _make_response(compressed, "text/html; charset=utf-8", "gzip")

        with patch("xcode.coding_agent.tools.webfetch.urlopen", return_value=response):
            result = web_tools["webfetch"].handler(
                {"url": "https://example.com", "format": "text"}
            )

        assert "Hello" in result
        assert "world" in result

    def test_json_is_returned_without_html_conversion(
        self, web_tools: dict[str, Any]
    ) -> None:
        payload = json.dumps({"key": "value"})
        response = _make_response(payload.encode("utf-8"), "application/json")

        with patch("xcode.coding_agent.tools.webfetch.urlopen", return_value=response):
            result = web_tools["webfetch"].handler(
                {"url": "https://api.example.com/data", "format": "markdown"}
            )

        assert result == payload

    def test_unsupported_image_content_type_raises(
        self, web_tools: dict[str, Any]
    ) -> None:
        response = _make_response(b"", "image/png")

        with (
            patch("xcode.coding_agent.tools.webfetch.urlopen", return_value=response),
            pytest.raises(ValueError, match="unsupported fetched image content type"),
        ):
            web_tools["webfetch"].handler({"url": "https://example.com/pic.png"})

    def test_http_error_raises(self, web_tools: dict[str, Any]) -> None:
        error = HTTPError(
            "https://example.com",
            500,
            "Internal Server Error",
            Message(),
            io.BytesIO(),
        )

        with (
            patch("xcode.coding_agent.tools.webfetch.urlopen", side_effect=error),
            pytest.raises(ValueError, match="HTTP 500"),
        ):
            web_tools["webfetch"].handler({"url": "https://example.com"})

    def test_format_aware_accept_header(self, web_tools: dict[str, Any]) -> None:
        html = "<html><body>text</body></html>"
        response = _make_response(html.encode("utf-8"), "text/html")

        with patch(
            "xcode.coding_agent.tools.webfetch.urlopen", return_value=response
        ) as mock_urlopen:
            web_tools["webfetch"].handler(
                {"url": "https://example.com", "format": "text"}
            )
            request = mock_urlopen.call_args[0][0]
            assert "text/plain;q=1.0" in request.headers["Accept"]


class TestWebSearch:
    def test_exa_sse_response_parsed(self, web_tools: dict[str, Any]) -> None:
        body = (
            "event: message\n"
            'data: {"result":{"content":[{"type":"text","text":"Search results here"}]}}\n'
        )
        response = _make_response(body.encode("utf-8"), "text/event-stream")

        with patch("xcode.coding_agent.tools.websearch.urlopen", return_value=response):
            result = web_tools["websearch"].handler(
                {"query": "test query", "numResults": 3}
            )

        assert "Search results here" in result

    def test_parallel_json_response_parsed(self, web_tools: dict[str, Any]) -> None:
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "Parallel results"}]},
            }
        )
        response = _make_response(body.encode("utf-8"), "application/json")

        with (
            patch(
                "xcode.coding_agent.tools.websearch._select_search_provider",
                return_value="parallel",
            ),
            patch("xcode.coding_agent.tools.websearch.urlopen", return_value=response),
        ):
            result = web_tools["websearch"].handler({"query": "test query"})

        assert "Parallel results" in result

    def test_empty_response_returns_default_message(
        self, web_tools: dict[str, Any]
    ) -> None:
        response = _make_response(b"", "text/event-stream")

        with patch("xcode.coding_agent.tools.websearch.urlopen", return_value=response):
            result = web_tools["websearch"].handler({"query": "test query"})

        assert "No search results found" in result

    def test_empty_query_raises(self, web_tools: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="query is required"):
            web_tools["websearch"].handler({"query": ""})
