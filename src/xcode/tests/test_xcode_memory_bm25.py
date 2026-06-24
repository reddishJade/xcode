from __future__ import annotations

import tempfile
from pathlib import Path

from rank_bm25 import BM25Okapi
from xcode.harness.memory import MemoryManager, MemorySearchEvalCase
import pytest


class TestBM25AndMemory:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def teardown_method(self, method) -> None:
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
        assert scores_apple[0] > scores_apple[1]
        assert scores_apple[2] == 0.0

        # 'fruit' matches both doc 0 and doc 1
        scores_fruit = bm25.get_scores(["fruit"])
        assert scores_fruit[0] > 0.0
        assert scores_fruit[1] > 0.0
        assert scores_fruit[2] == 0.0

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
        assert len(blocks) == 2
        assert blocks[0].startswith("## Incident 1")
        assert blocks[1].startswith("## Incident 2")

        # Test search
        results = manager.search_memory("tenacity connection timeout")
        assert len(results) == 1
        assert "Incident 2" in results[0]

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

        assert "Incident 2" in results[0]

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
        assert score_current > score_old

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

        assert success
        text = manager.memory_file.read_text(encoding="utf-8")
        assert "- Source: session-123" in text
        assert "- Scope: tasks" in text
        assert "- Confidence: 0.80" in text
        assert manager.validate_memory_block(text)
        trace_events = manager.drain_trace_events()
        assert [event.type for event in trace_events] == [
            "candidate_created",
            "accepted",
        ]

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

        assert all(result.passed for result in results)

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
        assert manager.validate_memory_block(valid_block)
        success = manager.add_memory_block(valid_block)
        assert success
        assert manager.memory_file.exists()
        assert "Incident 3" in manager.memory_file.read_text(encoding="utf-8")

        # 2. Invalid block (missing mandatory 'Takeaways' field)
        invalid_block = (
            "## Incident 4: Missing field block\n"
            "- Context/Query: Missing takeaways field\n"
            "- Solution: Fix it\n"
            "- Files: unknown.py\n"
        )
        assert not (manager.validate_memory_block(invalid_block))
        success_invalid = manager.add_memory_block(invalid_block)
        assert not (success_invalid)

        # Verify it was archived instead of appended
        archive_files = list(manager.archive_dir.glob("*.md"))
        assert len(archive_files) == 1
        assert "Incident 4" in archive_files[0].read_text(encoding="utf-8")
        trace_events = manager.drain_trace_events()
        assert trace_events[0].type == "candidate_created"
        assert trace_events[1].type == "accepted"
        assert trace_events[2].type == "candidate_created"
        assert trace_events[3].type == "rejected"
        assert trace_events[3].rejection_reason == "schema_validation_failed"

    def test_memory_trace_reports_supersede_and_lru_forget(self) -> None:
        manager = MemoryManager(self.root, max_blocks=1)
        first_block = (
            "## Incident A: Provider retry\n"
            "- Context/Query: Provider timeout\n"
            "- Solution: Add bounded retry\n"
            "- Files: provider.py\n"
            "- Takeaways: Retry only transient failures\n"
        )
        second_block = (
            "## Incident A: Provider retry\n"
            "- Context/Query: Provider timeout after refactor\n"
            "- Solution: Add bounded retry and jitter\n"
            "- Files: provider.py\n"
            "- Takeaways: Preserve the root cause\n"
        )
        third_block = (
            "## Incident B: Task lock\n"
            "- Context/Query: Task claim timeout\n"
            "- Solution: Use file lock retry\n"
            "- Files: task_store.py\n"
            "- Takeaways: Claiming must stay atomic\n"
        )

        assert manager.add_memory_block(first_block, source="repl")
        manager.drain_trace_events()

        assert manager.add_memory_block(second_block, source="repl")
        supersede_events = manager.drain_trace_events()
        assert [event.type for event in supersede_events] == [
            "candidate_created",
            "superseded",
            "accepted",
        ]

        assert manager.add_memory_block(third_block, source="repl")
        lru_events = manager.drain_trace_events()
        assert [event.type for event in lru_events] == [
            "candidate_created",
            "forgotten",
            "accepted",
        ]


if __name__ == "__main__":
    pytest.main()
