"""Web 检索与抓取工具。"""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from xcode.harness.skills import ToolInput, ToolSpec


USER_AGENT = "xcode-agent/1.0"
MAX_FETCH_BYTES = 5 * 1024 * 1024
MAX_SEARCH_BYTES = 256 * 1024


def build_web_tools() -> tuple[ToolSpec, ...]:
    """构建默认可用的 web 工具。"""

    def webfetch(data: ToolInput) -> str:
        url = _valid_url(str(data.get("url", "")).strip())
        output_format = _format(str(data.get("format", "markdown")).strip())
        timeout = _timeout(data.get("timeout"))
        raw, content_type, truncated = _fetch_url(url, timeout)
        text = raw.decode(_charset(content_type), errors="replace")
        if output_format == "html":
            return _with_truncation_notice(text, truncated)
        if output_format == "text":
            return _with_truncation_notice(_plain_text(text), truncated)
        return _with_truncation_notice(_markdown_text(text, url), truncated)

    def websearch(data: ToolInput) -> str:
        query = str(data.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        limit = _limit(data.get("numResults"))
        html = _fetch_url(
            f"https://duckduckgo.com/html/?q={quote_plus(query)}",
            _timeout(data.get("timeout")),
            max_bytes=MAX_SEARCH_BYTES,
        )[0].decode("utf-8", errors="replace")
        results = _duckduckgo_results(html, limit)
        return _render_search_results(query, results)

    return (
        ToolSpec(
            name="webfetch",
            description="Fetch a URL and return its content as markdown, text, or HTML.",
            input_hint='JSON: {"url":"https://example.com", "format":"markdown"}',
            handler=webfetch,
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
            read_only=True,
            group="core",
            prompt_snippet="Fetch web page content from an HTTP(S) URL.",
        ),
        ToolSpec(
            name="websearch",
            description="Search the web and return a compact list of result titles, URLs, and snippets.",
            input_hint='JSON: {"query":"python pathlib docs", "numResults":8}',
            handler=websearch,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "numResults": {"type": "integer", "minimum": 1, "maximum": 20},
                    "timeout": {"type": "number", "minimum": 1, "maximum": 120},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            read_only=True,
            group="core",
            prompt_snippet="Search the web for current external information.",
        ),
    )


def _fetch_url(
    url: str,
    timeout: float,
    *,
    max_bytes: int = MAX_FETCH_BYTES,
) -> tuple[bytes, str, bool]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/markdown, text/plain, text/html, application/json, application/xml, */*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            mime = _mime(content_type)
            if _is_image_mime(mime):
                raise ValueError(f"unsupported fetched image content type: {mime}")
            if not _is_textual_mime(mime):
                raise ValueError(f"unsupported fetched file content type: {mime}")
            raw = response.read(max_bytes + 1)
            return raw[:max_bytes], content_type, len(raw) > max_bytes
    except HTTPError as exc:
        raise ValueError(f"HTTP {exc.code} fetching {url}") from exc
    except URLError as exc:
        raise ValueError(f"failed to fetch {url}: {exc.reason}") from exc


def _valid_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an HTTP(S) URL")
    return url


def _timeout(value: object) -> float:
    if value is None:
        return 30.0
    if not isinstance(value, str | int | float):
        raise ValueError("timeout must be a number")
    timeout = float(value)
    if timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be between 0 and 120 seconds")
    return timeout


def _limit(value: object) -> int:
    if value is None:
        return 8
    if not isinstance(value, str | int):
        raise ValueError("limit must be an integer")
    limit = int(value)
    if limit < 1 or limit > 20:
        raise ValueError("numResults must be between 1 and 20")
    return limit


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


def _duckduckgo_results(html: str, limit: int) -> list[dict[str, str]]:
    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<url>[^"]+)">(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for match in pattern.finditer(html):
        results.append(
            {
                "title": _plain_text(match.group("title")),
                "url": match.group("url"),
                "snippet": _plain_text(match.group("snippet")),
            }
        )
        if len(results) >= limit:
            break
    return results


def _render_search_results(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return "No search results found. Please try a different query."
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                "",
                f"{index}. {result['title']}",
                f"   URL: {result['url']}",
                f"   Snippet: {result['snippet']}",
            ]
        )
    return "\n".join(lines)


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
