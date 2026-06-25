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

        # 没有 evidence / outcome 的 compaction 候选不能直接晋升为正式记忆
        assert not self.manager.memory_file.exists()
        candidate_files = list(self.manager.candidate_dir.glob("*.md"))
        assert len(candidate_files) == 1
        candidate_text = candidate_files[0].read_text(encoding="utf-8")
        assert "Incident 99: Memory compaction works" in candidate_text
        quarantine_files = list(self.manager.quarantine_dir.glob("*.md"))
        assert len(quarantine_files) == 1
        quarantine_text = quarantine_files[0].read_text(encoding="utf-8")
        assert "- Reason: evidence_gate_failed" in quarantine_text
        trace_events = self.manager.drain_trace_events()
        assert [event.type for event in trace_events] == [
            "candidate_created",
            "quarantined",
        ]
        assert trace_events[1].rejection_reason == "evidence_gate_failed"

    def test_consolidate_promotes_candidate_with_evidence(self) -> None:
        promotable_block = (
            "## Incident 100: Verified compaction flow\n"
            "- Context/Query: Compaction failure test\n"
            "- Solution: Persist candidate before promotion\n"
            "- Files: compaction.py\n"
            "- Takeaways: Promotion needs explicit evidence\n"
            "- Evidence: test:src/xcode/tests/test_xcode_memory_consolidation.py\n"
        )
        self.manager.consolidate(f"[Compressed]\n{promotable_block}")

        assert self.manager.memory_file.exists()
        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        assert "Incident 100: Verified compaction flow" in memory_text
        assert (
            "- Evidence: test:src/xcode/tests/test_xcode_memory_consolidation.py"
            in (memory_text)
        )
        candidate_files = list(self.manager.candidate_dir.glob("*.md"))
        assert len(candidate_files) == 1
        assert list(self.manager.quarantine_dir.glob("*.md")) == []
        trace_events = self.manager.drain_trace_events()
        assert [event.type for event in trace_events] == [
            "candidate_created",
            "accepted",
        ]


if __name__ == "__main__":
    pytest.main()
