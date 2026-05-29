from __future__ import annotations

import unittest
from typing import Any
from xcode.harness.agent_runtime.compaction import (
    budget_large_tool_outputs,
    LayeredCompactor,
)


class TestXcodeCompactionBudget(unittest.TestCase):
    def test_budget_no_truncation_when_under_threshold(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": "a" * 100}
                ],
            },
        ]

        # Use large thresholds so it's not triggered
        compacted = budget_large_tool_outputs(
            messages,
            large_tool_output_chars=50,
            large_tool_output_head_chars=10,
            large_tool_output_tail_chars=10,
            compact_token_threshold=10000,
            budget_trigger_token_ratio=0.5,
        )

        self.assertEqual(compacted[1]["content"][0]["content"], "a" * 100)

    def test_budget_truncation_when_over_threshold(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hello world test content"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "u1",
                        "content": "abcdefghijklmnopqrstuvwxyz",  # 26 chars
                    }
                ],
            },
        ]

        # Set trigger threshold extremely low so it triggers
        compacted = budget_large_tool_outputs(
            messages,
            large_tool_output_chars=10,
            large_tool_output_head_chars=5,
            large_tool_output_tail_chars=5,
            compact_token_threshold=5,
            budget_trigger_token_ratio=0.1,
        )

        content = compacted[1]["content"][0]["content"]
        self.assertTrue(content.startswith("abcde"))
        self.assertTrue(content.endswith("vwxyz"))
        self.assertIn("truncated 16 characters due to token budget", content)

    def test_layered_compactor_integration(self) -> None:
        # Verify that LayeredCompactor applies both stale snip and budgeting
        compactor = LayeredCompactor(
            large_tool_output_chars=10,
            large_tool_output_head_chars=5,
            large_tool_output_tail_chars=5,
            compact_token_threshold=5,
            budget_trigger_token_ratio=0.1,
            max_recent_messages=100,  # Avoid summarization
        )

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "u2",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "u1",
                        "content": "abcdefghijklmnopqrstuvwxyz",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "u2",
                        "content": "123456789012345",
                    },
                ],
            },
        ]

        compacted = compactor(messages)

        # The first tool result u1 should be snipped by stale_snip
        self.assertEqual(
            compacted[1]["content"][0]["content"],
            "[Content snipped - re-read if needed]",
        )

        # 最新一次 read_file 结果应保持完整，避免后续 Act 阶段基于陈旧内容执行。
        u2_content = compacted[1]["content"][1]["content"]
        self.assertEqual(u2_content, "123456789012345")

    def test_layered_compactor_budgets_non_preserved_tool_results(self) -> None:
        compactor = LayeredCompactor(
            large_tool_output_chars=10,
            large_tool_output_head_chars=5,
            large_tool_output_tail_chars=5,
            compact_token_threshold=5,
            budget_trigger_token_ratio=0.1,
            max_recent_messages=100,
        )

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "bash",
                        "input": {"command": "pytest"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "u1",
                        "content": "abcdefghijklmnopqrstuvwxyz",
                    }
                ],
            },
        ]

        compacted = compactor(messages)
        content = compacted[1]["content"][0]["content"]

        self.assertTrue(content.startswith("abcde"))
        self.assertTrue(content.endswith("vwxyz"))
        self.assertIn("truncated 16 characters due to token budget", content)


if __name__ == "__main__":
    unittest.main()
