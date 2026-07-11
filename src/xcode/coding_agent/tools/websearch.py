"""Web 搜索工具。"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from xcode.agent.types import ToolInput, ToolSpec


BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)
MAX_SEARCH_BYTES = 256 * 1024
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 120.0
MCP_EXA_URL = "https://mcp.exa.ai/mcp"
MCP_PARALLEL_URL = "https://search.parallel.ai/mcp"


def build_websearch_tool() -> ToolSpec:
    """构建 websearch 工具。"""

    def handler(
        data: ToolInput, _on_update: Callable[[str], None] | None = None
    ) -> str:
        query = str(data.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        limit = _limit(data.get("numResults"))
        timeout = _timeout(data.get("timeout"))
        livecrawl = _livecrawl(data.get("livecrawl"))
        search_type = _search_type(data.get("type"))
        provider = _select_search_provider()
        if provider == "parallel":
            text = _call_parallel_mcp(query, limit, timeout)
        else:
            text = _call_exa_mcp(query, limit, livecrawl, search_type, timeout)
        return text or "No search results found. Please try a different query."

    return ToolSpec(
        name="websearch",
        description="Search the web for current information. Supports optional result count, livecrawl mode, and search type.",
        input_hint='JSON: {"query":"python pathlib docs", "numResults":8}',
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "numResults": {"type": "integer", "minimum": 1, "maximum": 20},
                "livecrawl": {
                    "type": "string",
                    "enum": ["fallback", "preferred"],
                },
                "type": {
                    "type": "string",
                    "enum": ["auto", "fast", "deep"],
                },
                "timeout": {"type": "number", "minimum": 1, "maximum": 120},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        prompt_snippet="Search the web for current external information.",
    )


def _timeout(value: object) -> float:
    if value is None:
        return DEFAULT_TIMEOUT
    if not isinstance(value, str | int | float):
        raise ValueError("timeout must be a number")
    timeout = float(value)
    if timeout <= 0 or timeout > MAX_TIMEOUT:
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


def _livecrawl(value: object) -> Literal["fallback", "preferred"]:
    if value is None:
        return "fallback"
    if value in {"fallback", "preferred"}:
        return cast(Literal["fallback", "preferred"], value)
    raise ValueError("livecrawl must be fallback or preferred")


def _search_type(value: object) -> Literal["auto", "fast", "deep"]:
    if value is None:
        return "auto"
    if value in {"auto", "fast", "deep"}:
        return cast(Literal["auto", "fast", "deep"], value)
    raise ValueError("type must be auto, fast, or deep")


def _select_search_provider() -> Literal["exa", "parallel"]:
    if os.environ.get("OPENCODE_EXPERIMENTAL_PARALLEL") or os.environ.get(
        "PARALLEL_API_KEY"
    ):
        return "parallel"
    return "exa"


def _call_exa_mcp(
    query: str,
    num_results: int,
    livecrawl: Literal["fallback", "preferred"],
    search_type: Literal["auto", "fast", "deep"],
    timeout: float,
) -> str | None:
    url = MCP_EXA_URL
    api_key = os.environ.get("EXA_API_KEY")
    if api_key:
        url = f"{url}?exaApiKey={quote(api_key, safe='')}"
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": query,
                    "type": search_type,
                    "numResults": num_results,
                    "livecrawl": livecrawl,
                },
            },
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": BROWSER_UA,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_SEARCH_BYTES)
            text = raw.decode("utf-8", errors="replace")
            return _parse_mcp_response(text)
    except (HTTPError, URLError, OSError):
        return None


def _call_parallel_mcp(
    query: str,
    num_results: int,
    timeout: float,
) -> str | None:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {
                    "objective": query,
                    "search_queries": [query],
                    "session_id": "xcode",
                },
            },
        }
    ).encode("utf-8")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "pi/1.0",
    }
    api_key = os.environ.get("PARALLEL_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        MCP_PARALLEL_URL,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_SEARCH_BYTES)
            text = raw.decode("utf-8", errors="replace")
            return _parse_mcp_response(text)
    except (HTTPError, URLError, OSError):
        return None


def _parse_mcp_response(body: str) -> str | None:
    trimmed = body.strip()
    if not trimmed:
        return None
    try:
        parsed = json.loads(trimmed)
        content = parsed.get("result", {}).get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    return str(item["text"])
    except json.JSONDecodeError:
        pass
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                content = data.get("result", {}).get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("text"):
                            return str(item["text"])
            except json.JSONDecodeError:
                continue
    return None
