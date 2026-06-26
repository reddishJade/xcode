from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
from pathlib import Path

from rank_bm25 import BM25Okapi
from xcode.harness.memory import (
    MemoryEvidence,
    MemoryManager,
    MemorySearchEvalCase,
)
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

    def test_memory_search_prefers_exact_file_path_match(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Provider timeout generic\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry provider calls\n"
                "- Files: src/provider.py\n"
                "- Takeaways: Generic timeout memory\n"
                "\n"
                "## Task store lock\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry task store lock\n"
                "- Files: src/xcode/harness/task_store.py\n"
                "- Takeaways: Exact path should dominate\n"
            ),
            encoding="utf-8",
        )

        records = manager.search_memory_records(
            "src/xcode/harness/task_store.py",
            limit=2,
        )

        assert records[0].title == "Task store lock"

    def test_memory_search_prefers_exact_symbol_match(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Provider timeout generic\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry provider calls\n"
                "- Files: src/provider.py\n"
                "- Takeaways: Generic timeout memory\n"
                "\n"
                "## Snapshot procedure\n"
                "- Context/Query: Snapshot creation flow\n"
                "- Solution: Use SnapshotBuilder for reproducible snapshots\n"
                "- Files: src/xcode/harness/snapshot.py\n"
                "- Related-Symbols: SnapshotBuilder\n"
                "- Takeaways: Exact symbol should dominate\n"
            ),
            encoding="utf-8",
        )

        records = manager.search_memory_records("SnapshotBuilder", limit=2)

        assert records[0].title == "Snapshot procedure"

    def test_memory_search_supports_chinese_subphrase_match(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## 中文超时处理\n"
                "- Context/Query: 提供者连接超时重试\n"
                "- Solution: 使用有界退避重试\n"
                "- Files: provider.py\n"
                "- Takeaways: 超时重试需要保留根因\n"
            ),
            encoding="utf-8",
        )

        records = manager.search_memory_records("超时重试", limit=1)

        assert records
        assert records[0].title == "中文超时处理"

    def test_memory_candidate_retrieval_and_rerank_interfaces_preserve_order(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Provider timeout generic\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry provider calls\n"
                "- Files: src/provider.py\n"
                "- Takeaways: Generic timeout memory\n"
                "\n"
                "## Task store lock\n"
                "- Context/Query: Timeout retry failure\n"
                "- Solution: Retry task store lock\n"
                "- Files: src/xcode/harness/task_store.py\n"
                "- Takeaways: Exact path should dominate\n"
            ),
            encoding="utf-8",
        )

        candidates = manager.retrieve_memory_candidates(
            "src/xcode/harness/task_store.py",
        )
        ranked = manager.rerank_memory_candidates(
            candidates,
            "src/xcode/harness/task_store.py",
            limit=2,
        )

        assert len(candidates) == 2
        assert candidates[0].title == "Task store lock"
        assert ranked[0].title == "Task store lock"
        assert ranked[0].score >= candidates[0].score

    def test_custom_rerank_policy_can_disable_exact_path_bonus(self) -> None:
        from xcode.harness.memory import MemoryRerankPolicy

        block = (
            "## Task store lock\n"
            "- Context/Query: Timeout retry failure\n"
            "- Solution: Retry task store lock\n"
            "- Files: src/xcode/harness/task_store.py\n"
            "- Takeaways: Exact path bonus should be configurable\n"
        )
        default_manager = MemoryManager(self.root)
        default_manager.memory_file.write_text(block, encoding="utf-8")

        manager = MemoryManager(
            self.root,
            rerank_policy=MemoryRerankPolicy(
                exact_file_match_bonus=0.0,
                exact_basename_bonus=0.0,
                file_weight=0.1,
            ),
        )
        manager.memory_file.write_text(block, encoding="utf-8")

        record = manager.read_memory_records()[0]
        default_bonus = default_manager._exact_match_bonus(
            record,
            "src/xcode/harness/task_store.py",
        )
        custom_bonus = manager._exact_match_bonus(
            record,
            "src/xcode/harness/task_store.py",
        )

        assert default_bonus > 0.0
        assert custom_bonus == 0.0

    def test_memory_search_demotes_deprecated_records(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Old\n"
                "- Context/Query: timeout retry failure\n"
                "- Solution: obsolete fix\n"
                "- Files: old.py\n"
                "- Takeaways: old path\n"
                "- Status: deprecated\n"
                "- Confidence: 0.80\n"
                "\n"
                "## Current\n"
                "- Context/Query: timeout retry failure\n"
                "- Solution: current fix\n"
                "- Files: current.py\n"
                "- Takeaways: current path\n"
                "- Confidence: 0.80\n"
            ),
            encoding="utf-8",
        )

        records = manager.search_memory_records("timeout retry failure", limit=2)

        assert records[0].title == "Current"
        assert records[0].score > records[1].score

    def test_memory_search_demotes_stale_records(self) -> None:
        manager = MemoryManager(self.root)
        now = datetime.now(timezone.utc)
        stale = (now - timedelta(days=180)).isoformat()
        fresh = (now - timedelta(days=2)).isoformat()
        manager.memory_file.write_text(
            (
                "## Stale timeout fix\n"
                "- Context/Query: provider timeout retry failure\n"
                "- Solution: old retry plan\n"
                "- Files: provider.py\n"
                "- Takeaways: stale memory\n"
                f"- Modified: {stale}\n"
                "\n"
                "## Fresh timeout fix\n"
                "- Context/Query: provider timeout retry failure\n"
                "- Solution: current retry plan\n"
                "- Files: provider.py\n"
                "- Takeaways: fresh memory\n"
                f"- Modified: {fresh}\n"
            ),
            encoding="utf-8",
        )

        records = manager.search_memory_records(
            "provider timeout retry failure",
            limit=2,
        )

        assert records[0].title == "Fresh timeout fix"
        assert records[0].score > records[1].score

    def test_legacy_record_gets_stable_id_and_inferred_type(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Incident: Provider timeout\n"
                "- Context/Query: Provider timeout during deploy\n"
                "- Solution: Retry with backoff\n"
                "- Files: provider.py\n"
                "- Takeaways: Treat this as an incident\n"
            ),
            encoding="utf-8",
        )

        record = manager.read_memory_records()[0]

        assert record.memory_id.startswith("mem_")
        assert record.memory_type == "episodic"
        assert record.status == "active"
        assert record.validity == "unknown"

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
            memory_type="procedural",
            status="active",
            validity="verified",
            evidence=(
                MemoryEvidence(
                    "test", "pytest src/xcode/tests/test_xcode_memory_bm25.py"
                ),
                MemoryEvidence("file", "src/xcode/harness/memory/manager.py"),
            ),
        )

        assert success
        text = manager.memory_file.read_text(encoding="utf-8")
        assert "- Memory-ID: mem_" in text
        assert "- Memory-Type: procedural" in text
        assert "- Source: session-123" in text
        assert "- Source-Session: session-123" in text
        assert "- Scope: tasks" in text
        assert "- Confidence: 0.80" in text
        assert "- Status: active" in text
        assert "- Validity: verified" in text
        assert (
            "- Evidence: test:pytest src/xcode/tests/test_xcode_memory_bm25.py; "
            "file:src/xcode/harness/memory/manager.py"
        ) in text
        assert manager.validate_memory_block(text)
        record = manager.read_memory_records()[0]
        assert record.memory_type == "procedural"
        assert record.source_session == "session-123"
        assert len(record.evidence) == 2
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

    def test_retention_keeps_high_value_semantic_record_over_weaker_new_episode(self) -> None:
        manager = MemoryManager(self.root, max_blocks=1)
        stable_fact = (
            "## Architecture fact\n"
            "- Context/Query: Shared runtime constraint\n"
            "- Solution: Keep provider wiring in assembly.py\n"
            "- Files: assembly.py\n"
            "- Takeaways: Stable architecture facts should survive weak churn\n"
        )
        weak_episode = (
            "## Incident weak retry\n"
            "- Context/Query: Retry tweak during one failing run\n"
            "- Solution: Add one more retry\n"
            "- Files: provider.py\n"
            "- Takeaways: This needs review before reuse\n"
        )

        assert manager.add_memory_block(
            stable_fact,
            validity="verified",
            memory_type="semantic",
        )
        record = manager.read_memory_records(layer="project")[0]
        manager.record_adopted_records((record,))
        manager.record_session_outcome("success")
        manager.drain_trace_events()

        assert manager.add_memory_block(
            weak_episode,
            status="needs_review",
            validity="needs_review",
        )

        records = manager.read_memory_records(layer="project")
        assert len(records) == 1
        assert records[0].title == "Architecture fact"
        events = manager.drain_trace_events()
        assert [event.type for event in events] == [
            "candidate_created",
            "forgotten",
            "accepted",
        ]
        assert events[1].title == "Incident weak retry"

    def test_migrate_legacy_records_rewrites_missing_identity_metadata(self) -> None:
        manager = MemoryManager(self.root)
        manager.memory_file.write_text(
            (
                "## Team convention\n"
                "- Context/Query: Shared architecture fact\n"
                "- Solution: Keep provider wiring in assembly.py\n"
                "- Files: src/xcode/harness/assembly.py\n"
                "- Takeaways: Stable runtime facts belong in memory\n"
            ),
            encoding="utf-8",
        )

        updated_count = manager.migrate_legacy_records()

        assert updated_count == 1
        rewritten = manager.memory_file.read_text(encoding="utf-8")
        assert "- Memory-ID: mem_" in rewritten
        assert "- Memory-Type: semantic" in rewritten
        assert "- Status: active" in rewritten
        assert "- Validity: unknown" in rewritten

    def test_session_outcome_tracks_exposure_without_counting_helpfulness(self) -> None:
        manager = MemoryManager(self.root)
        manager.add_memory_block(
            (
                "## Provider retry playbook\n"
                "- Context/Query: Provider timeout retry\n"
                "- Solution: Retry transient provider failures\n"
                "- Files: provider.py\n"
                "- Takeaways: Keep bounded retries\n"
            )
        )
        manager.drain_trace_events()

        record = manager.search_memory_records(
            "provider timeout retry",
            source="prompt",
        )[0]
        manager.record_injected_records((record,))

        updated = manager.record_session_outcome("success")

        assert updated == 1
        persisted = manager.read_memory_records(layer="project")[0]
        assert persisted.retrieval_count == 1
        assert persisted.injection_count == 1
        assert persisted.adoption_count == 0
        assert persisted.success_count == 0
        assert persisted.failure_count == 0
        assert persisted.utility == 0.0
        assert persisted.last_outcome == "success"

    def test_adopted_feedback_updates_utility_and_review_state(self) -> None:
        manager = MemoryManager(self.root)
        manager.add_memory_block(
            (
                "## Task lock workaround\n"
                "- Context/Query: Task lock timeout\n"
                "- Solution: Retry file lock with backoff\n"
                "- Files: task_store.py\n"
                "- Takeaways: Treat lock contention as transient\n"
            ),
            validity="verified",
        )
        manager.drain_trace_events()

        record = manager.read_memory_records(layer="project")[0]
        manager.record_adopted_records((record,))
        manager.record_session_outcome("failure")

        failed = manager.read_memory_records(layer="project")[0]
        assert failed.adoption_count == 1
        assert failed.failure_count == 1
        assert failed.success_count == 0
        assert failed.utility == -1.0
        assert failed.status == "needs_review"
        assert failed.validity == "needs_review"

        manager.record_adopted_records((failed,))
        manager.record_session_outcome("success")

        recovered = manager.read_memory_records(layer="project")[0]
        assert recovered.adoption_count == 2
        assert recovered.failure_count == 1
        assert recovered.success_count == 1
        assert recovered.utility == 0.0
        assert recovered.status == "active"
        assert recovered.validity == "verified"

    def test_explicit_reference_tracks_title_without_promoting_to_adoption(self) -> None:
        manager = MemoryManager(self.root)
        manager.add_memory_block(
            (
                "## Provider retry playbook\n"
                "- Context/Query: Provider timeout retry\n"
                "- Solution: Retry transient provider failures\n"
                "- Files: provider.py\n"
                "- Takeaways: Keep bounded retries\n"
            )
        )
        manager.drain_trace_events()

        record = manager.search_memory_records("provider timeout retry", source="prompt")[
            0
        ]
        manager.record_injected_records((record,))

        matched = manager.record_explicit_references(
            "Following Provider retry playbook for this task."
        )
        updated = manager.record_session_outcome("success")

        assert matched == 1
        assert updated == 1
        persisted = manager.read_memory_records(layer="project")[0]
        assert persisted.reference_count == 1
        assert persisted.adoption_count == 0
        assert persisted.success_count == 0
        assert persisted.utility == 0.0

    def test_successful_episodic_records_promote_procedural_candidate(self) -> None:
        from xcode.harness.memory.parsing import with_metadata

        manager = MemoryManager(self.root)
        first = (
            "## Incident A: Provider timeout retry\n"
            "- Context/Query: Provider timeout during deploy\n"
            "- Solution: Retry transient provider failures with backoff\n"
            "- Files: provider.py\n"
            "- Takeaways: Keep bounded retries and preserve the root cause\n"
        )
        second = (
            "## Incident B: Provider timeout retry\n"
            "- Context/Query: Provider timeout during batch sync\n"
            "- Solution: Retry transient provider failures with backoff\n"
            "- Files: provider.py\n"
            "- Takeaways: Keep bounded retries and preserve the root cause\n"
        )
        manager.memory_file.write_text(
            (
                with_metadata(
                    first,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
                + "\n\n"
                + with_metadata(
                    second,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
            ),
            encoding="utf-8",
        )
        manager.drain_trace_events()

        records = manager.read_memory_records(layer="project")
        manager.record_adopted_records((records[0],))
        manager.record_session_outcome("success")
        refreshed = manager.read_memory_records(layer="project")
        manager.record_adopted_records((refreshed[1],))
        updated = manager.record_session_outcome("success")

        assert updated == 1
        candidate_files = list(manager.candidate_dir.glob("*.md"))
        assert len(candidate_files) == 1
        candidate_text = candidate_files[0].read_text(encoding="utf-8")
        assert "## How to: Retry transient provider failures with backoff" in candidate_text
        assert "- Memory-Type: procedural" in candidate_text
        assert "- Source-Records: mem_" in candidate_text
        assert "- Evidence: memory:mem_" in candidate_text

    def test_counterexample_quarantines_procedural_candidate(self) -> None:
        from xcode.harness.memory.parsing import with_metadata

        manager = MemoryManager(self.root)
        successful = (
            "## Incident A: Task lock retry\n"
            "- Context/Query: Task lock timeout in worker A\n"
            "- Solution: Retry file lock with backoff\n"
            "- Files: task_store.py\n"
            "- Takeaways: Treat lock contention as transient\n"
        )
        failing = (
            "## Incident B: Task lock retry\n"
            "- Context/Query: Task lock timeout in worker B\n"
            "- Solution: Retry file lock with backoff\n"
            "- Files: task_store.py\n"
            "- Takeaways: Treat lock contention as transient\n"
        )
        extra_success = (
            "## Incident C: Task lock retry\n"
            "- Context/Query: Task lock timeout in worker C\n"
            "- Solution: Retry file lock with backoff\n"
            "- Files: task_store.py\n"
            "- Takeaways: Treat lock contention as transient\n"
        )
        manager.memory_file.write_text(
            (
                with_metadata(
                    successful,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
                + "\n\n"
                + with_metadata(
                    failing,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
                + "\n\n"
                + with_metadata(
                    extra_success,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
            ),
            encoding="utf-8",
        )
        manager.drain_trace_events()

        records = manager.read_memory_records(layer="project")
        manager.record_adopted_records((records[0],))
        manager.record_session_outcome("success")
        refreshed = manager.read_memory_records(layer="project")
        manager.record_adopted_records((refreshed[1],))
        manager.record_session_outcome("failure")
        refreshed = manager.read_memory_records(layer="project")
        manager.record_adopted_records((refreshed[2],))
        manager.record_session_outcome("success")

        assert list(manager.candidate_dir.glob("*.md")) == []
        quarantine_files = list(manager.quarantine_dir.glob("*.md"))
        assert len(quarantine_files) == 1
        quarantine_text = quarantine_files[0].read_text(encoding="utf-8")
        assert "- Reason: counterexample_gate_failed" in quarantine_text
        assert "- Counterexamples: mem_" in quarantine_text

    def test_promote_candidate_moves_procedural_candidate_into_formal_memory(self) -> None:
        from xcode.harness.memory.parsing import with_metadata

        manager = MemoryManager(self.root)
        first = (
            "## Incident A: Provider timeout retry\n"
            "- Context/Query: Provider timeout during deploy\n"
            "- Solution: Retry transient provider failures with backoff\n"
            "- Files: provider.py\n"
            "- Takeaways: Keep bounded retries and preserve the root cause\n"
        )
        second = (
            "## Incident B: Provider timeout retry\n"
            "- Context/Query: Provider timeout during batch sync\n"
            "- Solution: Retry transient provider failures with backoff\n"
            "- Files: provider.py\n"
            "- Takeaways: Keep bounded retries and preserve the root cause\n"
        )
        manager.memory_file.write_text(
            (
                with_metadata(
                    first,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
                + "\n\n"
                + with_metadata(
                    second,
                    layer="project",
                    source=None,
                    scope=None,
                    confidence=None,
                    validity="verified",
                )
            ),
            encoding="utf-8",
        )
        records = manager.read_memory_records(layer="project")
        manager.record_adopted_records((records[0],))
        manager.record_session_outcome("success")
        refreshed = manager.read_memory_records(layer="project")
        manager.record_adopted_records((refreshed[1],))
        manager.record_session_outcome("success")

        promoted = manager.promote_candidate(
            "How to: Retry transient provider failures with backoff"
        )

        assert promoted
        assert list(manager.candidate_dir.glob("*.md")) == []
        records = manager.read_memory_records(layer="project")
        procedural = next(
            record
            for record in records
            if record.title == "How to: Retry transient provider failures with backoff"
        )
        assert procedural.memory_type == "procedural"
        assert procedural.status == "active"
        assert procedural.validity == "derived"

    def test_reject_candidate_moves_file_to_quarantine(self) -> None:
        manager = MemoryManager(self.root)
        manager._write_candidate_block(
            (
                "## How to: Review provider retry\n"
                "- Context/Query: Abstracted from successful incidents\n"
                "- Solution: Review provider retry\n"
                "- Files: provider.py\n"
                "- Takeaways: Keep retries bounded\n"
                "- Memory-Type: procedural\n"
            ),
            layer="project",
            source="test",
        )

        rejected = manager.reject_candidate(
            "How to: Review provider retry",
            reason="manual_rejection",
        )

        assert rejected
        assert list(manager.candidate_dir.glob("*.md")) == []
        quarantine_files = list(manager.quarantine_dir.glob("*.md"))
        assert len(quarantine_files) == 1
        quarantine_text = quarantine_files[0].read_text(encoding="utf-8")
        assert "- Reason: manual_rejection" in quarantine_text
        assert "## How to: Review provider retry" in quarantine_text


if __name__ == "__main__":
    pytest.main()
