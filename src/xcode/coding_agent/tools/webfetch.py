"""URL 抓取工具。"""

from __future__ import annotations

import gzip
import re
from html import unescape
from html.parser import HTMLParser
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from xcode.agent.types import ToolSpec


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)
MAX_FETCH_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 120.0


def build_webfetch_tool() -> ToolSpec:
    """构建 webfetch 工具。"""

    def handler(data: ToolInput) -> str:
        url = _valid_url(str(data.get("url", "")).strip())
        output_format = _format(str(data.get("format", "markdown")).strip())
        timeout = _timeout(data.get("timeout"))
        raw, content_type, truncated = _fetch_url(url, timeout, output_format)
        text = raw.decode(_charset(content_type), errors="replace")
        if not _is_html_mime(_mime(content_type)):
            return _with_truncation_notice(text, truncated)
        if output_format == "html":
            return _with_truncation_notice(text, truncated)
        if output_format == "text":
            return _with_truncation_notice(_plain_text(text), truncated)
        return _with_truncation_notice(_markdown_text(text, url), truncated)

    return ToolSpec(
        name="webfetch",
        description="Fetch content from an HTTP or HTTPS URL and return it as text, markdown, or HTML.",
        input_hint='JSON: {"url":"https://example.com", "format":"markdown"}',
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP(S) URL to fetch."},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "text", "html"],
                },
                "timeout": {"type": "number", "minimum": 1, "maximum": 120},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        prompt_snippet="Fetch web page content from an HTTP(S) URL.",
    )


def _fetch_url(
    url: str,
    timeout: float,
    output_format: Literal["markdown", "text", "html"] = "markdown",
    *,
    max_bytes: int = MAX_FETCH_BYTES,
) -> tuple[bytes, str, bool]:
    request = Request(
        url,
        headers={
            "User-Agent": BROWSER_UA,
            "Accept": _accept_header(output_format),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            content_encoding = response.headers.get("content-encoding")
            mime = _mime(content_type)
            if _is_image_mime(mime):
                raise ValueError(f"unsupported fetched image content type: {mime}")
            if not _is_textual_mime(mime):
                raise ValueError(f"unsupported fetched file content type: {mime}")
            raw = response.read()
            raw = _decompress(raw, content_encoding)
            return raw[:max_bytes], content_type, len(raw) > max_bytes
    except HTTPError as exc:
        raise ValueError(f"HTTP {exc.code} fetching {url}") from exc
    except URLError as exc:
        raise ValueError(f"failed to fetch {url}: {exc.reason}") from exc


def _decompress(raw: bytes, content_encoding: str | None) -> bytes:
    if not content_encoding:
        return raw
    encoding = content_encoding.lower()
    if "gzip" in encoding:
        return gzip.decompress(raw)
    if "deflate" in encoding:
        import zlib

        return zlib.decompress(raw)
    return raw


def _accept_header(output_format: Literal["markdown", "text", "html"]) -> str:
    if output_format == "markdown":
        return (
            "text/markdown;q=1.0, text/x-markdown;q=0.9, "
            "text/plain;q=0.8, text/html;q=0.7, */*;q=0.1"
        )
    if output_format == "text":
        return "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1"
    return (
        "text/html;q=1.0, application/xhtml+xml;q=0.9, "
        "text/plain;q=0.8, text/markdown;q=0.7, */*;q=0.1"
    )


def _valid_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an HTTP(S) URL")
    return url


def _timeout(value: object) -> float:
    if value is None:
        return DEFAULT_TIMEOUT
    if not isinstance(value, str | int | float):
        raise ValueError("timeout must be a number")
    timeout = float(value)
    if timeout <= 0 or timeout > MAX_TIMEOUT:
        raise ValueError("timeout must be between 0 and 120 seconds")
    return timeout


def _format(value: str) -> Literal["markdown", "text", "html"]:
    if value in {"markdown", "text", "html"}:
        return cast(Literal["markdown", "text", "html"], value)
    raise ValueError("format must be markdown, text, or html")


def _charset(content_type: str) -> str:
    match = re.search(r"charset=([^;]+)", content_type, re.IGNORECASE)
    return match.group(1).strip() if match else "utf-8"


def _mime(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_image_mime(mime: str) -> bool:
    return mime.startswith("image/") and mime != "image/svg+xml"


def _is_textual_mime(mime: str) -> bool:
    return (
        not mime
        or mime.startswith("text/")
        or mime in {"application/json", "application/xml", "application/javascript"}
        or mime.endswith("+json")
        or mime.endswith("+xml")
    )


def _is_html_mime(mime: str) -> bool:
    return mime in {"text/html", "application/xhtml+xml"} or mime.endswith("+html")


def _plain_text(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    return "\n".join(line for line in parser.text().splitlines() if line.strip())


def _markdown_text(html: str, url: str) -> str:
    text = _html_to_markdown(html)
    return f"Source: {url}\n\n{text}"


def _with_truncation_notice(text: str, truncated: bool) -> str:
    if not truncated:
        return text
    return text.rstrip() + "\n\n[Output truncated at 5 MB.]"


def _html_to_markdown(html: str) -> str:
    cleaned = re.sub(
        r"<(script|style|noscript|iframe|object|embed)[^>]*>[\s\S]*?</\1>",
        "",
        html,
        flags=re.IGNORECASE,
    )
    replacements = (
        (r"<h1[^>]*>(.*?)</h1>", r"# \1\n\n"),
        (r"<h2[^>]*>(.*?)</h2>", r"## \1\n\n"),
        (r"<h3[^>]*>(.*?)</h3>", r"### \1\n\n"),
        (r"<strong[^>]*>(.*?)</strong>", r"**\1**"),
        (r"<b[^>]*>(.*?)</b>", r"**\1**"),
        (r"<em[^>]*>(.*?)</em>", r"*\1*"),
        (r"<i[^>]*>(.*?)</i>", r"*\1*"),
        (r"<code[^>]*>(.*?)</code>", r"`\1`"),
        (r"<li[^>]*>(.*?)</li>", r"- \1\n"),
        (r"<p[^>]*>(.*?)</p>", r"\1\n\n"),
        (r"<br\s*/?>", "\n"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(
        r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        r"[\2](\1)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    return unescape(re.sub(r"\n{4,}", "\n\n\n", cleaned).strip())


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return "\n".join(self._parts)
