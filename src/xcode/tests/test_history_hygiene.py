"""历史修复和 hygiene 测试。

测试工具配对修复和请求 hygiene 压缩。
"""

import unittest

from xcode.agent.history import (
    apply_request_hygiene,
    repair_tool_pairing,
)
from xcode.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from xcode.agent.types import TextContent, ToolCallContent, ToolResultContent


class TestRepairToolPairing(unittest.TestCase):
    """测试工具调用配对修复。"""

    def test_repair_removes_orphan_results(self):
        """测试移除孤儿 tool_result。"""
        messages = [
            UserMessage(content=[TextContent(text="test")]),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="orphan_id",
                        content="orphan result",
                    )
                ],
                is_error=False,
            ),
        ]
        repaired = repair_tool_pairing(messages)
        # 孤儿 result 应被移除
        self.assertEqual(len(repaired), 1)
        self.assertIsInstance(repaired[0], UserMessage)

    def test_repair_removes_incomplete_calls(self):
        """测试移除未完成的 tool_call。"""
        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="incomplete_id",
                        name="test_tool",
                        arguments={},
                    )
                ],
            ),
            UserMessage(content=[TextContent(text="test")]),
        ]
        repaired = repair_tool_pairing(messages)
        # 未完成 call 的 assistant 消息应被移除
        self.assertEqual(len(repaired), 1)
        self.assertIsInstance(repaired[0], UserMessage)

    def test_repair_keeps_valid_pairs(self):
        """测试保留有效配对。"""
        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="valid_id",
                        name="test_tool",
                        arguments={},
                    )
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="valid_id",
                        content="result",
                    )
                ],
                is_error=False,
            ),
        ]
        repaired = repair_tool_pairing(messages)
        self.assertEqual(len(repaired), 2)
        self.assertIsInstance(repaired[0], AssistantMessage)
        self.assertIsInstance(repaired[1], ToolResultMessage)

    def test_repair_mixed_content(self):
        """测试混合内容（text + tool_call）。"""
        messages = [
            AssistantMessage(
                content=[
                    TextContent(text="Let me check"),
                    ToolCallContent(
                        id="call_1",
                        name="tool_a",
                        arguments={},
                    ),
                    ToolCallContent(
                        id="incomplete",
                        name="tool_b",
                        arguments={},
                    ),
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="call_1",
                        content="result",
                    )
                ],
                is_error=False,
            ),
        ]
        repaired = repair_tool_pairing(messages)
        self.assertEqual(len(repaired), 2)
        # assistant 消息应保留 text + 有效 call
        assistant = repaired[0]
        self.assertIsInstance(assistant, AssistantMessage)
        self.assertEqual(len(assistant.content), 2)
        self.assertIsInstance(assistant.content[0], TextContent)
        self.assertIsInstance(assistant.content[1], ToolCallContent)


class TestApplyRequestHygiene(unittest.TestCase):
    """测试请求 hygiene 压缩。"""

    def test_hygiene_truncates_large_result(self):
        """测试压缩超大 tool_result。"""
        large_content = "\n".join([f"line {i}" for i in range(200)])
        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="call_1",
                        name="read_file",
                        arguments={},
                    )
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="call_1",
                        content=large_content,
                    )
                ],
                is_error=False,
            ),
        ]
        cleaned = apply_request_hygiene(
            messages,
            keep_head_lines=10,
            keep_tail_lines=10,
        )
        result_msg = cleaned[1]
        self.assertIsInstance(result_msg, ToolResultMessage)
        result_content = result_msg.content[0].content
        # 应包含省略标记
        self.assertIn("omitted", result_content)
        # 行数应减少
        self.assertLess(len(result_content.splitlines()), 200)

    def test_hygiene_truncates_tool_args(self):
        """测试压缩超长工具参数。"""
        long_string = "x" * 2000
        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="call_1",
                        name="write_file",
                        arguments={"content": long_string},
                    )
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="call_1",
                        content="success",
                    )
                ],
                is_error=False,
            ),
        ]
        cleaned = apply_request_hygiene(messages, max_tool_arg_length=100)
        assistant = cleaned[0]
        self.assertIsInstance(assistant, AssistantMessage)
        call = assistant.content[0]
        self.assertIsInstance(call, ToolCallContent)
        # 参数应被压缩
        self.assertIn("truncated", str(call.arguments.get("content")))

    def test_hygiene_detects_base64(self):
        """测试检测 base64 payload。"""
        base64_content = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 10
        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="call_1",
                        name="read_file",
                        arguments={},
                    )
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="call_1",
                        content=base64_content,
                    )
                ],
                is_error=False,
            ),
        ]
        cleaned = apply_request_hygiene(messages)
        result_msg = cleaned[1]
        result_content = result_msg.content[0].content
        # 应替换为占位符
        self.assertIn("base64", result_content)
        self.assertIn("bytes", result_content)

    def test_hygiene_preserves_signal_lines(self):
        """测试保留错误/警告信号行。"""
        lines = []
        lines.extend([f"normal line {i}" for i in range(50)])
        lines.append("ERROR: something went wrong")
        lines.extend([f"normal line {i}" for i in range(50, 100)])
        content = "\n".join(lines)

        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="call_1",
                        name="test",
                        arguments={},
                    )
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="call_1",
                        content=content,
                    )
                ],
                is_error=False,
            ),
        ]
        cleaned = apply_request_hygiene(
            messages,
            keep_head_lines=10,
            keep_tail_lines=10,
        )
        result_msg = cleaned[1]
        result_content = result_msg.content[0].content
        # 应保留 ERROR 行
        self.assertIn("ERROR: something went wrong", result_content)

    def test_hygiene_no_truncate_small_content(self):
        """测试小内容不压缩。"""
        small_content = "just a few lines\nof normal output"
        messages = [
            AssistantMessage(
                content=[
                    ToolCallContent(
                        id="call_1",
                        name="test",
                        arguments={"param": "short"},
                    )
                ],
            ),
            ToolResultMessage(
                content=[
                    ToolResultContent(
                        tool_use_id="call_1",
                        content=small_content,
                    )
                ],
                is_error=False,
            ),
        ]
        cleaned = apply_request_hygiene(messages)
        # 应保持不变
        self.assertEqual(cleaned[1].content[0].content, small_content)
        self.assertEqual(
            cleaned[0].content[0].arguments.get("param"),
            "short",
        )


if __name__ == "__main__":
    unittest.main()
