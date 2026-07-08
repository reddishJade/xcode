"""Agent 辅助函数纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.agent_helpers import (
    _tool_result_status,
    _tool_result_text,
    text_from_blocks,
)
from xcode.agent.messages import ToolResultMessage
from xcode.agent.types import TextContent


class TestToolResultStatus:
    def test_ok(self) -> None:
        msg = ToolResultMessage(tool_call_id="c1", tool_name="t", content="ok", is_error=False)
        assert _tool_result_status(msg) == "ok"

    def test_error(self) -> None:
        msg = ToolResultMessage(tool_call_id="c1", tool_name="t", content="failed", is_error=True)
        assert _tool_result_status(msg) == "error"

    def test_interrupted(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c1", tool_name="t", content="Interrupted by user", is_error=True
        )
        assert _tool_result_status(msg) == "interrupted"

    def test_cancelled(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="c1", tool_name="t", content="cancelled", is_error=True
        )
        assert _tool_result_status(msg) == "interrupted"


class TestToolResultText:
    def test_string(self) -> None:
        assert _tool_result_text("hello") == "hello"

    def test_list_with_text_content(self) -> None:
        content = [TextContent(text="hello"), TextContent(text=" world")]
        assert _tool_result_text(content) == "hello world"

    def test_empty_list(self) -> None:
        assert _tool_result_text([]) == ""


class TestTextFromBlocks:
    def test_text_type_blocks(self) -> None:
        blocks = [{"type": "text", "text": "Hello"}, {"type": "text", "text": " World"}]
        assert text_from_blocks(blocks) == "Hello World"

    def test_blocks_with_text_key(self) -> None:
        blocks = [{"text": "Hello"}, {"text": "World"}]
        result = text_from_blocks(blocks)
        assert "Hello" in result

    def test_non_text_blocks_skipped(self) -> None:
        blocks = [{"type": "tool_use", "name": "search"}, {"type": "text", "text": "result"}]
        assert text_from_blocks(blocks) == "result"

    def test_empty(self) -> None:
        assert text_from_blocks([]) == ""
