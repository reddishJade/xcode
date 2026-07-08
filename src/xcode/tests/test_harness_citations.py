"""Citation marker 装饰器纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.prompting.citations import (
    decorate_citable_messages,
    _get_citation_sources,
    _line_numbered_text,
    citation_sources_as_dicts,
)
from xcode.agent.messages import ToolResultMessage, UserMessage
from xcode.agent.types import CitationSource, CITATION_SOURCES_METADATA_KEY


class TestDecorateCitableMessages:
    def test_no_citation_sources(self) -> None:
        msgs = [UserMessage(content="hello")]
        result = decorate_citable_messages(msgs)
        assert len(result) == 1

    def test_with_citation_sources(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c1",
            tool_name="read_file",
            content="line1\nline2",
            metadata={CITATION_SOURCES_METADATA_KEY: [
                {"kind": "file", "path": "src/main.py", "start_line": 1, "end_line": 2, "text": "content"},
            ]},
        )
        result = decorate_citable_messages([msg])
        assert len(result) == 1
        text = str(result[0].content)
        assert "cite" in text
        assert "turn0file0" in text or "\ue200" in text


class TestGetCitationSources:
    def test_empty_metadata(self) -> None:
        msg = ToolResultMessage(tool_call_id="c1", tool_name="t", content="x")
        assert _get_citation_sources(msg) == []

    def test_with_sources(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            content="x",
            metadata={CITATION_SOURCES_METADATA_KEY: [
                {"kind": "file", "path": "a.py", "start_line": 1, "end_line": 5, "text": "code"},
            ]},
        )
        sources = _get_citation_sources(msg)
        assert len(sources) == 1
        assert sources[0].path == "a.py"


class TestLineNumberedText:
    def test_adds_line_numbers(self) -> None:
        sources = [CitationSource(kind="file", path="a.py", start_line=10, end_line=12, text="")]
        result = _line_numbered_text("hello\nworld\n!", sources)
        assert "[L10]" in result
        assert "[L11]" in result
        assert "[L12]" in result

    def test_non_file_no_numbers(self) -> None:
        sources = [CitationSource(kind="search", path="", start_line=1, end_line=1, text="")]
        result = _line_numbered_text("hello", sources)
        assert "[L1]" not in result

    def test_empty_sources_no_numbers(self) -> None:
        result = _line_numbered_text("hello", [])
        assert "[L1]" not in result


class TestCitationSourcesAsDicts:
    def test_converts(self) -> None:
        sources = [CitationSource(kind="file", path="a.py", start_line=1, end_line=5, text="code")]
        result = citation_sources_as_dicts(sources)
        assert len(result) == 1
        assert result[0]["path"] == "a.py"
