"""Citation marker decorator for local file/search tool results.

在 provider 边界为可引用的工具输出添加 citation marker 头 + 行号文本。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from xcode.agent.messages import AgentMessage, ToolResultMessage
from xcode.agent.types import TextContent
from xcode.harness.skills import CITATION_SOURCES_METADATA_KEY, CitationSource


def decorate_citable_messages(messages: list[AgentMessage]) -> list[AgentMessage]:
    """扫描消息列表中的可引用工具结果，添加 citation marker。

    按 prompt 顺序分配 turn ID，在每个可引用工具结果文本前插入
    marker 头 + 行号标识。
    """
    turn_index = 0
    decorated: list[AgentMessage] = []
    for msg in messages:
        if isinstance(msg, ToolResultMessage) and _has_citation_sources(msg):
            sources = _get_citation_sources(msg)
            if sources:
                decorated_turn = _decorate_tool_result(msg, turn_index, sources)
                decorated.append(decorated_turn)
                turn_index += 1
                continue
        decorated.append(msg)
    return decorated


def _has_citation_sources(msg: ToolResultMessage) -> bool:
    if not msg.metadata:
        return False
    raw = msg.metadata.get(CITATION_SOURCES_METADATA_KEY)
    return bool(raw)


def _get_citation_sources(msg: ToolResultMessage) -> list[CitationSource]:
    if not msg.metadata:
        return []
    raw = msg.metadata.get(CITATION_SOURCES_METADATA_KEY)
    if isinstance(raw, list):
        return [_dict_to_citation_source(item) for item in raw]
    return []


def _dict_to_citation_source(item: object) -> CitationSource:
    if isinstance(item, dict):
        return CitationSource(
            kind=item.get("kind", "file"),
            path=str(item.get("path", "")),
            start_line=int(item.get("start_line", 1)),
            end_line=int(item.get("end_line", 1)),
            text=str(item.get("text", "")),
        )
    if isinstance(item, CitationSource):
        return item
    return CitationSource(
        kind="file", path="", start_line=1, end_line=1, text=str(item)
    )


def _decorate_tool_result(
    msg: ToolResultMessage,
    turn_index: int,
    sources: list[CitationSource],
) -> ToolResultMessage:
    """为单个工具结果的消息内容添加 citation marker 前缀。"""
    original_text = _get_tool_result_text(msg)
    parts: list[str] = []
    for source_index, source in enumerate(sources):
        kind_abbr = "file" if source.kind == "file" else "search"
        source_id = f"turn{turn_index}{kind_abbr}{source_index}"
        locator = _locator_text(source)
        marker = f"\ue200cite\ue202{source_id}\ue202{locator}\ue201"
        parts.append(f"Citation Marker: {marker}\n")
        parts.append(f"Path: {source.path}\n")
        parts.append(f"Lines: L{source.start_line}-L{source.end_line}\n\n")

    line_numbered = _line_numbered_text(original_text, sources)
    if parts:
        citation_header = "".join(parts)
        new_text = citation_header + line_numbered
    else:
        new_text = line_numbered

    return ToolResultMessage(
        tool_call_id=msg.tool_call_id,
        tool_name=msg.tool_name,
        content=new_text,
        is_error=msg.is_error,
        metadata=msg.metadata,
    )


def _locator_text(source: CitationSource) -> str:
    if source.kind == "file":
        if source.start_line == source.end_line:
            return f"L{source.start_line}"
        return f"L{source.start_line}-L{source.end_line}"
    return f"L{source.start_line}"


def _get_tool_result_text(msg: ToolResultMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        else:
            parts.append(str(item))
    return "".join(parts)


def _line_numbered_text(
    text: str,
    sources: list[CitationSource],
) -> str:
    """对于 file 类源，为可见行添加 [Lxx] 前缀。"""
    if not sources or sources[0].kind != "file":
        return text
    lines = text.splitlines()
    numbered: list[str] = []
    start_line = sources[0].start_line
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered.append(f"[L{line_num}] {line}")
    return "\n".join(numbered)


def citation_sources_as_dicts(sources: list[CitationSource]) -> list[dict[str, Any]]:
    return [asdict(s) for s in sources]
