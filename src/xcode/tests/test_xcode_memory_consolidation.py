from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xcode.harness.agent_runtime.compaction import LayeredCompactor
from xcode.harness.memory import MemoryManager


class TestMemoryConsolidationHook(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manager = MemoryManager(self.root)

    def tearDown(self) -> None:
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
        self.assertTrue(len(final_messages) < len(messages))
        self.assertEqual(final_messages[1]["role"], "user")
        self.assertTrue(final_messages[1]["content"].startswith("[Compressed]"))

        # Verify that the memory block was consolidated into MEMORY.md!
        self.assertTrue(self.manager.memory_file.exists())
        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        self.assertIn("Incident 99: Memory compaction works", memory_text)
        self.assertIn("Hook into Summary Compact", memory_text)

    def test_consolidation_hook_archives_invalid_attempt(self) -> None:
        compactor = LayeredCompactor(
            max_recent_messages=2, on_compact=self.manager.consolidate
        )

        # Malformed block (missing 'Takeaways' field)
        invalid_block = (
            "## Incident 100: Corrupt compaction\n"
            "- Context/Query: Compaction failure test\n"
            "- Solution: Archive it\n"
            "- Files: compaction.py\n"
        )

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Normal user message"},
            {"role": "assistant", "content": invalid_block},
            {"role": "user", "content": "Latest user message"},
            {"role": "assistant", "content": "Latest assistant reply"},
        ]

        # Trigger compaction
        compactor(messages)

        # Verify that MEMORY.md does not contain the invalid block
        if self.manager.memory_file.exists():
            self.assertNotIn(
                "Incident 100", self.manager.memory_file.read_text(encoding="utf-8")
            )

        # Verify that the invalid block was safely archived to the archive directory
        archive_files = list(self.manager.archive_dir.glob("*.md"))
        self.assertEqual(len(archive_files), 1)
        archive_text = archive_files[0].read_text(encoding="utf-8")
        self.assertIn("Incident 100: Corrupt compaction", archive_text)


if __name__ == "__main__":
    unittest.main()
