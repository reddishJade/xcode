"""消息历史修复与请求卫生单元测试。"""

from __future__ import annotations

from xcode.agent._hygiene import (
    _is_base64_payload,
    _is_signal_line,
    _truncate_tool_args,
    _truncate_tool_result,
    apply_request_hygiene,
    repair_tool_pairing,
)
from xcode.agent.messages import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import ToolCallContent


class TestRepairToolPairing:
    def test_empty_messages(self) -> None:
        assert repair_tool_pairing([]) == []

    def test_removes_unmatched_tool_calls(self) -> None:
        msgs = [
            AssistantMessage(
                content=[
                    ToolCallContent(id="c1", name="search"),
                    ToolCallContent(id="c2", name="not_done"),
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(tool_call_id="c1", tool_name="search", content="ok"),
        ]
        result = repair_tool_pairing(msgs)
        assert len(result) == 2
        remaining = result[0].content
        assert len(remaining) == 1
        assert remaining[0].id == "c1"

    def test_removes_unmatched_tool_results(self) -> None:
        msgs = [
            AssistantMessage(
                content=[ToolCallContent(id="c1", name="search")],
                stop_reason="tool_use",
            ),
            ToolResultMessage(tool_call_id="c2", tool_name="other", content="orphan"),
            AssistantMessage(
                content=[ToolCallContent(id="c2", name="other")],
                stop_reason="tool_use",
            ),
        ]
        result = repair_tool_pairing(msgs)
        # c1 call has no result → removed; c2 result + call are a valid pair
        assert len(result) == 2
        assert result[0].tool_call_id == "c2"


class TestApplyRequestHygiene:
    def test_truncates_long_tool_result(self) -> None:
        long_content = "line1\nline2\nerror: something\nline3\n" * 20
        msgs = [
            AssistantMessage(
                content=[ToolCallContent(id="c1", name="bash")],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="bash",
                content=long_content,
                is_error=False,
            ),
        ]
        result = apply_request_hygiene(msgs, keep_head_lines=2, keep_tail_lines=2)
        tool_result = result[1]
        content = str(tool_result.content)
        assert "omitted" in content or len(content) < len(long_content)

    def test_truncates_long_tool_args(self) -> None:
        long_arg = "x" * 2000
        msgs = [
            AssistantMessage(
                content=[
                    ToolCallContent(id="c1", name="write", arguments={"data": long_arg})
                ],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c1", tool_name="write", content="done", is_error=False
            ),
        ]
        result = apply_request_hygiene(msgs, max_tool_arg_length=200)
        block = result[0].content[0]
        assert "<truncated" in str(block.arguments["data"])

    def test_short_content_unchanged(self) -> None:
        msgs = [
            UserMessage(content="short"),
        ]
        result = apply_request_hygiene(msgs)
        assert result[0].content == "short"

    def test_keeps_signal_lines_in_truncated_output(self) -> None:
        content = "\n".join([f"line{i}" for i in range(200)])
        content += "\nTraceback: error occurred\n"
        content += "\n".join([f"line{i}" for i in range(200, 400)])
        msgs = [
            AssistantMessage(
                content=[ToolCallContent(id="c1", name="bash")],
                stop_reason="tool_use",
            ),
            ToolResultMessage(
                tool_call_id="c1", tool_name="bash", content=content, is_error=False
            ),
        ]
        result = apply_request_hygiene(msgs, keep_head_lines=3, keep_tail_lines=3)
        result_text = str(result[1].content)
        assert "Traceback" in result_text


class TestTruncateToolArgs:
    def test_truncates_long_string_value(self) -> None:
        args = {"data": "x" * 500}
        result = _truncate_tool_args(args, max_length=100)
        assert "<truncated" in result["data"]  # type: ignore[operator]

    def test_short_value_unchanged(self) -> None:
        args = {"data": "short"}
        result = _truncate_tool_args(args, max_length=100)
        assert result["data"] == "short"

    def test_nested_dict(self) -> None:
        args = {"outer": {"inner": "x" * 500}}
        result = _truncate_tool_args(args, max_length=100)
        assert "<truncated" in str(result["outer"])


class TestTruncateToolResult:
    def test_base64_detected(self) -> None:
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ+/=" * 20
        result = _truncate_tool_result(content, 100, 5, 5)
        assert "base64" in result

    def test_short_content_unchanged(self) -> None:
        result = _truncate_tool_result("hello", 1000, 50, 50)
        assert result == "hello"

    def test_long_content_truncated(self) -> None:
        lines = [f"line{i}" for i in range(100)]
        result = _truncate_tool_result("\n".join(lines), 10000, 5, 5)
        assert "omitted" in result


class TestIsBase64Payload:
    def test_detects_base64(self) -> None:
        assert _is_base64_payload("A" * 200)

    def test_short_text_not_base64(self) -> None:
        assert not _is_base64_payload("hello")

    def test_plain_text_not_base64(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 10
        assert not _is_base64_payload(text)


class TestIsSignalLine:
    def test_detects_error(self) -> None:
        assert _is_signal_line("Error: something broke")

    def test_detects_exception(self) -> None:
        assert _is_signal_line("Exception: KeyError")

    def test_ordinary_line(self) -> None:
        assert not _is_signal_line("Everything is fine")
