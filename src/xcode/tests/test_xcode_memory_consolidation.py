from __future__ import annotations

import tempfile
from pathlib import Path

from xcode.harness.agent_runtime.compaction import LayeredCompactor
from xcode.harness.memory import MemoryManager
import pytest


class TestMemoryConsolidationHook:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manager = MemoryManager(self.root)

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_consolidation_hook_triggers_on_compaction(self) -> None:
        # Create a compactor with 2 recent messages max to force compaction
        compactor = LayeredCompactor(
            max_recent_messages=2, on_compact=self.manager.consolidate
        )

        # Build a message list that has more than 3 messages (max_recent + 1)
        # We inject a memory block inside the user/assistant content to see if it is consolidated!
        valid_block = (
            "## Incident 99: Memory compaction works\n"
            "- Context/Query: Compaction test runs\n"
            "- Solution: Hook into Summary Compact\n"
            "- Files: compaction.py\n"
            "- Takeaways: Clean and elegant\n"
        )

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Normal user message"},
            {"role": "assistant", "content": valid_block},
            {"role": "user", "content": "Latest user message"},
            {"role": "assistant", "content": "Latest assistant reply"},
        ]

        # Trigger compaction
        final_messages = compactor(messages)

        # Compactor should summarize the older messages (index 1 to 2)
        # Check that the summary message was created and starts with [Compressed]
        assert len(final_messages) < len(messages)
        assert final_messages[1]["role"] == "user"
        assert final_messages[1]["content"].startswith("[Compressed]")

        assert self.manager.memory_file.exists()
        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        assert "Incident 99: Memory compaction works" in memory_text
        trace_events = self.manager.drain_trace_events()
        assert [event.type for event in trace_events] == [
            "candidate_created",
            "accepted",
        ]
        assert trace_events[1].title == "Incident 99: Memory compaction works"

    def test_consolidate_rejects_ephemeral_session_only_memory(self) -> None:
        ephemeral_block = (
            "## Incident 100: Session-only compaction flow\n"
            "- Context/Query: This session only compaction failure test\n"
            "- Solution: Persist current turn scratch state\n"
            "- Files: compaction.py\n"
            "- Takeaways: Temporary notes from this turn should not become memory\n"
        )
        self.manager.consolidate(f"[Compressed]\n{ephemeral_block}")

        assert not self.manager.memory_file.exists()
        archive_files = list(self.manager.archive_dir.glob("*.md"))
        assert len(archive_files) == 1
        archive_text = archive_files[0].read_text(encoding="utf-8")
        assert "Incident 100: Session-only compaction flow" in archive_text
        trace_events = self.manager.drain_trace_events()
        assert [event.type for event in trace_events] == [
            "rejected",
        ]
        assert trace_events[0].rejection_reason == "scope_gate_failed"


if __name__ == "__main__":
    pytest.main()
