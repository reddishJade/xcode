"""重复工具调用检测测试。

测试带文件变更感知的重复工具调用抑制。
"""

import unittest

from xcode.agent.types import ToolCallContent
from xcode.agent.watchdog import (
    RepeatDetector,
    is_file_mutation_tool,
    is_file_read_tool,
    should_clear_read_history,
    tool_call_signature,
    tool_calls_signature,
)


class TestToolSignature(unittest.TestCase):
    """测试工具签名生成。"""

    def test_single_call_signature(self):
        """测试单个工具调用签名。"""
        call = ToolCallContent(
            id="call_1",
            name="read_file",
            arguments={"path": "/test/file.txt"},
        )
        sig = tool_call_signature(call)
        self.assertIn("read_file", sig)
        self.assertIn("/test/file.txt", sig)

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
        self.assertEqual(tool_call_signature(call1), tool_call_signature(call2))

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
        self.assertEqual(tool_calls_signature(calls1), tool_calls_signature(calls2))


class TestToolClassification(unittest.TestCase):
    """测试工具分类。"""

    def test_file_mutation_tools(self):
        """测试文件变更工具识别。"""
        self.assertTrue(is_file_mutation_tool("write_file"))
        self.assertTrue(is_file_mutation_tool("edit_file"))
        self.assertTrue(is_file_mutation_tool("bash"))
        self.assertFalse(is_file_mutation_tool("read_file"))

    def test_file_read_tools(self):
        """测试只读工具识别。"""
        self.assertTrue(is_file_read_tool("read_file"))
        self.assertTrue(is_file_read_tool("grep_search"))
        self.assertTrue(is_file_read_tool("glob_files"))
        self.assertFalse(is_file_read_tool("write_file"))

    def test_should_clear_read_history(self):
        """测试是否应清除只读历史。"""
        read_calls = [
            ToolCallContent(id="1", name="read_file", arguments={}),
        ]
        write_calls = [
            ToolCallContent(id="2", name="write_file", arguments={}),
        ]
        # 只读调用不清除历史
        self.assertFalse(should_clear_read_history(read_calls, []))
        # 写入调用清除历史
        self.assertTrue(should_clear_read_history(write_calls, []))


class TestRepeatDetector(unittest.TestCase):
    """测试重复检测器。"""

    def test_no_repeat_different_calls(self):
        """测试不同调用不触发重复。"""
        detector = RepeatDetector(limit=3)
        calls1 = [ToolCallContent(id="1", name="read_file", arguments={"path": "a"})]
        calls2 = [ToolCallContent(id="2", name="read_file", arguments={"path": "b"})]

        is_repeat, _ = detector.check_and_update(calls1)
        self.assertFalse(is_repeat)
        is_repeat, _ = detector.check_and_update(calls2)
        self.assertFalse(is_repeat)

    def test_repeat_same_calls(self):
        """测试相同调用触发重复限制。"""
        detector = RepeatDetector(limit=3)
        calls = [ToolCallContent(id="1", name="read_file", arguments={"path": "a"})]

        # 第 1、2 次不触发
        is_repeat, _ = detector.check_and_update(calls)
        self.assertFalse(is_repeat)
        is_repeat, _ = detector.check_and_update(calls)
        self.assertFalse(is_repeat)

        # 第 3 次触发
        is_repeat, reason = detector.check_and_update(calls)
        self.assertTrue(is_repeat)
        self.assertIn("重复", reason)

    def test_clear_history_after_mutation(self):
        """测试文件变更后清除只读历史。"""
        detector = RepeatDetector(limit=3)

        # 读文件 3 次
        read_calls = [
            ToolCallContent(id="1", name="read_file", arguments={"path": "a"})
        ]
        detector.check_and_update(read_calls)
        detector.check_and_update(read_calls)
        detector.check_and_update(read_calls)

        # 写文件（清除历史）
        write_calls = [
            ToolCallContent(id="2", name="write_file", arguments={"path": "a"})
        ]
        is_repeat, _ = detector.check_and_update(write_calls)
        self.assertFalse(is_repeat)

        # 再次读文件不应触发重复（历史已清除）
        is_repeat, _ = detector.check_and_update(read_calls)
        self.assertFalse(is_repeat)

    def test_reset(self):
        """测试重置检测器。"""
        detector = RepeatDetector(limit=3)
        calls = [ToolCallContent(id="1", name="test", arguments={})]

        detector.check_and_update(calls)
        detector.check_and_update(calls)
        detector.reset()

        # 重置后重新计数
        is_repeat, _ = detector.check_and_update(calls)
        self.assertFalse(is_repeat)

    def test_empty_calls(self):
        """测试空调用列表。"""
        detector = RepeatDetector(limit=3)
        is_repeat, _ = detector.check_and_update([])
        self.assertFalse(is_repeat)


if __name__ == "__main__":
    unittest.main()
