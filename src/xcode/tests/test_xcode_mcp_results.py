"""MCP 现代工具结果映射测试。"""

from __future__ import annotations

import unittest

from xcode.agent.types import FileContent, ImageContent
from xcode.harness.mcp.results import (
    MCP_RESULT_METADATA_KEY,
    convert_mcp_tool_result,
)
from xcode.harness.skills import AGENT_CONTENT_BLOCKS_METADATA_KEY


class McpToolResultTests(unittest.TestCase):
    """验证 MCP result 到宿主结果模型的转换。"""

    def test_validates_and_preserves_structured_content(self) -> None:
        """structuredContent 会按 outputSchema 校验并进入详情。"""
        output = convert_mcp_tool_result(
            {
                "content": [],
                "structuredContent": {
                    "temperature": 22.5,
                    "conditions": "clear",
                },
            },
            {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "conditions": {"type": "string"},
                },
                "required": ["temperature", "conditions"],
                "additionalProperties": False,
            },
        )

        self.assertFalse(output.is_error)
        self.assertIn("[MCP structuredContent]", output)
        details = output.metadata[MCP_RESULT_METADATA_KEY]
        self.assertIsInstance(details, dict)
        assert isinstance(details, dict)
        self.assertEqual(
            details["structuredContent"],
            {"temperature": 22.5, "conditions": "clear"},
        )
        self.assertEqual(details["validation"]["status"], "valid")

    def test_invalid_structured_content_is_a_tool_error(self) -> None:
        """不符合 outputSchema 的结果保留内容并标记为错误。"""
        output = convert_mcp_tool_result(
            {
                "content": [{"type": "text", "text": "server text"}],
                "structuredContent": {"count": "wrong"},
            },
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        )

        self.assertTrue(output.is_error)
        self.assertIn("server text", output)
        self.assertIn("violates outputSchema", output)
        details = output.metadata[MCP_RESULT_METADATA_KEY]
        assert isinstance(details, dict)
        self.assertEqual(details["validation"]["status"], "invalid")

    def test_maps_all_supported_non_text_blocks_and_continues(self) -> None:
        """非文本块进入宿主类型，未知块不丢失后续内容。"""
        output = convert_mcp_tool_result(
            {
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image",
                        "data": "image-data",
                        "mimeType": "image/png",
                        "annotations": {"audience": ["user"]},
                    },
                    {
                        "type": "audio",
                        "data": "audio-data",
                        "mimeType": "audio/wav",
                    },
                    {
                        "type": "resource_link",
                        "uri": "file:///project/main.py",
                        "name": "main.py",
                        "mimeType": "text/x-python",
                    },
                    {
                        "type": "resource",
                        "resource": {
                            "uri": "file:///project/readme.txt",
                            "mimeType": "text/plain",
                            "text": "embedded",
                        },
                    },
                    {"type": "future_content", "value": 3},
                    {"type": "text", "text": "after"},
                ]
            },
            None,
        )

        self.assertTrue(output.is_error)
        self.assertIn("before", output)
        self.assertIn("after", output)
        self.assertIn("future_content", output)
        blocks = output.metadata[AGENT_CONTENT_BLOCKS_METADATA_KEY]
        self.assertIsInstance(blocks, list)
        assert isinstance(blocks, list)
        self.assertEqual(len(blocks), 4)
        self.assertIsInstance(blocks[0], ImageContent)
        self.assertTrue(all(isinstance(block, FileContent) for block in blocks[1:]))

        image = blocks[0]
        assert isinstance(image, ImageContent)
        self.assertEqual(image.source["annotations"], {"audience": ["user"]})
        resource = blocks[3]
        assert isinstance(resource, FileContent)
        self.assertEqual(resource.filename, "readme.txt")
        self.assertEqual(resource.file_data, "embedded")

    def test_reports_complete_redacted_unsupported_block(self) -> None:
        """unsupported 诊断保留完整块，同时脱敏凭据。"""
        output = convert_mcp_tool_result(
            {
                "content": [
                    {
                        "type": "future",
                        "token": "Bearer secret-value",
                        "nested": {"value": 7},
                    }
                ]
            },
            None,
        )

        self.assertIn('"nested": {"value": 7}', output)
        self.assertIn("Bearer ****", output)
        self.assertNotIn("secret-value", output)

    def test_requires_structured_content_when_output_schema_exists(self) -> None:
        """声明 outputSchema 后缺少 structuredContent 会标记协议错误。"""
        output = convert_mcp_tool_result(
            {"content": [{"type": "text", "text": "legacy only"}]},
            {"type": "object"},
        )

        self.assertTrue(output.is_error)
        self.assertIn("returned no structuredContent", output)

    def test_preserves_server_error_status(self) -> None:
        """MCP isError 会直接进入宿主错误状态。"""
        output = convert_mcp_tool_result(
            {
                "content": [{"type": "text", "text": "request rejected"}],
                "isError": True,
            },
            {"type": "object"},
        )

        self.assertTrue(output.is_error)
        self.assertEqual(output, "request rejected")
        details = output.metadata[MCP_RESULT_METADATA_KEY]
        assert isinstance(details, dict)
        self.assertEqual(details["validation"]["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
