"""重复工具调用检测测试。

测试带文件变更感知的重复工具调用抑制。
"""

from xcode.agent.types import ToolCallContent
from xcode.agent.watchdog import (
is_file_mutation_tool,
    is_file_read_tool,
    should_clear_read_history,
    tool_call_signature,
    tool_calls_signature,
)
import pytest
class TestToolSignature:
    """测试工具签名生成。"""

    def test_single_call_signature(self):
        """测试单个工具调用签名。"""
        call = ToolCallContent(
            id="call_1",
            name="read_file",
            arguments={"path": "/test/file.txt"},
        )
        sig = tool_call_signature(call)
        assert "read_file" in sig
        assert "/test/file.txt" in sig

    def test_signature_stable_same_args(self):
        """测试相同参数生成相同签名。"""
        call1 = ToolCallContent(
            id="call_1",
            name="test",
            arguments={"a": 1, "b": 2},
        )
        call2 = ToolCallContent(
            id="call_2",
            name="test",
            arguments={"b": 2, "a": 1},  # 参数顺序不同
        )
        assert tool_call_signature(call1) == tool_call_signature(call2)

    def test_batch_signature_order_independent(self):
        """测试批次签名与调用顺序无关。"""
        calls1 = [
            ToolCallContent(id="1", name="tool_a", arguments={}),
            ToolCallContent(id="2", name="tool_b", arguments={}),
        ]
        calls2 = [
            ToolCallContent(id="3", name="tool_b", arguments={}),
            ToolCallContent(id="4", name="tool_a", arguments={}),
        ]
        assert tool_calls_signature(calls1) == tool_calls_signature(calls2)

class TestToolClassification:
    """测试工具分类。"""

    def test_file_mutation_tools(self):
        """测试文件变更工具识别。"""
        assert is_file_mutation_tool("write_file")
        assert is_file_mutation_tool("edit_file")
        assert is_file_mutation_tool("bash")
        assert not (is_file_mutation_tool("read_file"))

    def test_file_read_tools(self):
        """测试只读工具识别。"""
        assert is_file_read_tool("read_file")
        assert is_file_read_tool("grep_search")
        assert is_file_read_tool("glob_files")
        assert not (is_file_read_tool("write_file"))

    def test_should_clear_read_history(self):
        """测试是否应清除只读历史。"""
        read_calls = [
            ToolCallContent(id="1", name="read_file", arguments={}),
        ]
        write_calls = [
            ToolCallContent(id="2", name="write_file", arguments={}),
        ]
        # 只读调用不清除历史
        assert not (should_clear_read_history(read_calls, []))
        # 写入调用清除历史
        assert should_clear_read_history(write_calls, [])

if __name__ == "__main__":
    pytest.main()
