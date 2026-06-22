from __future__ import annotations

from typing import Any
from xcode.harness.agent_runtime.compaction import stale_snip_file_reads
import pytest
class TestXcodeStaleSnip:
    def test_stale_snip_only_keeps_latest_read_file(self) -> None:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "Read files",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "use_b1",
                        "name": "read_file",
                        "input": {"path": "/workspace/b.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "use_a1",
                        "name": "read_file",
                        "input": {"path": "/workspace/a.txt"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "use_b1",
                        "content": "content of b version 1",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "use_a1",
                        "content": "content of a",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "use_b2",
                        "name": "read_file",
                        "input": {"path": "/workspace/b.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "use_other",
                        "name": "grep_search",
                        "input": {"query": "something"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "use_b2",
                        "content": "content of b version 2",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "use_other",
                        "content": "some grep results",
                    },
                ],
            },
        ]

        compacted = stale_snip_file_reads(messages)

        # Assertions
        # 1. Check that the first read of b.txt is snipped
        b1_result = compacted[2]["content"][0]
        assert b1_result["tool_use_id"] == "use_b1"
        assert b1_result["content"] == "[Content snipped - re-read if needed]"

        # 2. Check that the read of a.txt is NOT snipped
        a_result = compacted[2]["content"][1]
        assert a_result["tool_use_id"] == "use_a1"
        assert a_result["content"] == "content of a"

        # 3. Check that the second read of b.txt is NOT snipped
        b2_result = compacted[4]["content"][0]
        assert b2_result["tool_use_id"] == "use_b2"
        assert b2_result["content"] == "content of b version 2"

        # 4. Check that grep_search is NOT snipped
        other_result = compacted[4]["content"][1]
        assert other_result["tool_use_id"] == "use_other"
        assert other_result["content"] == "some grep results"

    def test_stale_snip_no_error_on_missing_fields(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "u1", "name": "read_file"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": "foo"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u2",
                        "name": "read_file",
                        "input": {"path": "/w/a.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u2", "content": "bar"}
                ],
            },
        ]
        # Should execute successfully without throwing errors
        compacted = stale_snip_file_reads(messages)
        assert len(compacted) == 5

if __name__ == "__main__":
    pytest.main()
