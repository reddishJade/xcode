from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rank_bm25 import BM25Okapi
from xcode.harness.memory import MemoryManager, MemorySearchEvalCase


class TestBM25AndMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bm25_okapi_calculations(self) -> None:
        corpus = [
            ["apple", "banana", "fruit"],
            ["banana", "cherry", "fruit"],
            ["dog", "cat", "pet"],
        ]
        bm25 = BM25Okapi(corpus)

        # 'apple' should match the first document strongly
        scores_apple = bm25.get_scores(["apple"])
        self.assertTrue(scores_apple[0] > scores_apple[1])
        self.assertEqual(scores_apple[2], 0.0)

        # 'fruit' matches both doc 0 and doc 1
        scores_fruit = bm25.get_scores(["fruit"])
        self.assertTrue(scores_fruit[0] > 0.0)
        self.assertTrue(scores_fruit[1] > 0.0)
        self.assertEqual(scores_fruit[2], 0.0)

    def test_memory_manager_read_and_search(self) -> None:
        manager = MemoryManager(self.root)

        # Prepare a MEMORY.md file with 2 blocks
        memory_content = (
            "## Incident 1: Fix index crash\n"
            "- Context/Query: Index out of bounds\n"
            "- Solution: Add guard checks\n"
            "- Files: index.py\n"
            "- Takeaways: Always validate indexes\n"
            "\n"
            "## Incident 2: Timeout retry failure\n"
            "- Context/Query: Subprocess connection timeout\n"
            "- Solution: Integrate tenacity retries\n"
            "- Files: factory.py\n"
            "- Takeaways: Identify transient errors\n"
        )
        manager.memory_file.write_text(memory_content, encoding="utf-8")

        # Test read
        blocks = manager.read_memory_blocks()
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0].startswith("## Incident 1"))
        self.assertTrue(blocks[1].startswith("## Incident 2"))

        # Test search
        results = manager.search_memory("tenacity connection timeout")
        self.assertEqual(len(results), 1)
        self.assertIn("Incident 2", results[0])

    def test_memory_search_prefers_matching_scope(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Incident 1: Timeout in provider\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry provider calls\n"
                "- Files: core/providers/factory.py\n"
                "- Takeaways: Provider timeouts need retry\n"
                "- Scope: providers\n"
                "- Confidence: 0.60\n"
                "\n"
                "## Incident 2: Timeout in tasks\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry task claim\n"
                "- Files: harness/task_store.py\n"
                "- Takeaways: Task timeouts need lock retry\n"
                "- Scope: tasks\n"
                "- Confidence: 0.60\n"
            ),
            encoding="utf-8",
        )

        results = manager.search_memory("timeout retry", scope="tasks", limit=2)

        self.assertIn("Incident 2", results[0])

    def test_memory_search_demotes_deprecated_records(self) -> None:
        from xcode.harness.memory.parsing import adjust_score, parse_memory_record

        base = 1.0
        old = parse_memory_record(
            "## Old\n- Context/Query: test\n- Status: deprecated\n- Confidence: 0.80\n"
        )
        current = parse_memory_record(
            "## Current\n- Context/Query: test\n- Confidence: 0.80\n"
        )
        score_old = adjust_score(base, old, "test", None)
        score_current = adjust_score(base, current, "test", None)
        self.assertGreater(score_current, score_old)

    def test_add_memory_block_appends_metadata_without_breaking_contract(self) -> None:
        manager = MemoryManager(self.root)
        block = (
            "## Incident 3: Git lock error\n"
            "- Context/Query: Git lock timeout\n"
            "- Solution: Release lock on exit\n"
            "- Files: tasks.py\n"
            "- Takeaways: filelock handles this automatically\n"
        )

        success = manager.add_memory_block(
            block,
            source="session-123",
            scope="tasks",
            confidence=0.8,
        )

        self.assertTrue(success)
        text = manager.memory_file.read_text(encoding="utf-8")
        self.assertIn("- Source: session-123", text)
        self.assertIn("- Scope: tasks", text)
        self.assertIn("- Confidence: 0.80", text)
        self.assertTrue(manager.validate_memory_block(text))

    def test_memory_search_eval_reports_topk_hit(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Incident 1: Provider retry\n"
                "- Context/Query: Provider timeout retry\n"
                "- Solution: Retry provider calls\n"
                "- Files: core/providers/factory.py\n"
                "- Takeaways: Provider timeouts need retry\n"
                "\n"
                "## Incident 2: Task lock\n"
                "- Context/Query: Task file lock conflict\n"
                "- Solution: Use exclusive directory lock\n"
                "- Files: harness/task_store.py\n"
                "- Takeaways: Task claiming needs atomic lock\n"
            ),
            encoding="utf-8",
        )

        results = manager.evaluate_search(
            [
                MemorySearchEvalCase(
                    query="provider timeout",
                    expected_title_contains="Provider retry",
                ),
                MemorySearchEvalCase(
                    query="task lock",
                    expected_title_contains="Task lock",
                ),
            ]
        )

        self.assertTrue(all(result.passed for result in results))

    def test_memory_manager_validation_and_archive(self) -> None:
        manager = MemoryManager(self.root)

        # 1. Valid block
        valid_block = (
            "## Incident 3: Git lock error\n"
            "- Context/Query: Git lock timeout\n"
            "- Solution: Release lock on exit\n"
            "- Files: tasks.py\n"
            "- Takeaways: filelock handles this automatically\n"
        )
        self.assertTrue(manager.validate_memory_block(valid_block))
        success = manager.add_memory_block(valid_block)
        self.assertTrue(success)
        self.assertTrue(manager.memory_file.exists())
        self.assertIn("Incident 3", manager.memory_file.read_text(encoding="utf-8"))

        # 2. Invalid block (missing mandatory 'Takeaways' field)
        invalid_block = (
            "## Incident 4: Missing field block\n"
            "- Context/Query: Missing takeaways field\n"
            "- Solution: Fix it\n"
            "- Files: unknown.py\n"
        )
        self.assertFalse(manager.validate_memory_block(invalid_block))
        success_invalid = manager.add_memory_block(invalid_block)
        self.assertFalse(success_invalid)

        # Verify it was archived instead of appended
        archive_files = list(manager.archive_dir.glob("*.md"))
        self.assertEqual(len(archive_files), 1)
        self.assertIn("Incident 4", archive_files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
