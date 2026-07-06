"""Regression tests for evidence-aware memory feedback."""

from __future__ import annotations

from pathlib import Path

from xcode.harness.memory import MemoryManager, MemoryRecord


def _block() -> str:
    return (
        "## Provider retry policy\n"
        "- Context/Query: Provider timeout retry handling\n"
        "- Solution: Retry transient provider failures with bounded backoff.\n"
        "- Files: src/xcode/provider.py\n"
        "- Takeaways: Preserve the original failure reason.\n"
    )


def _retrieve_and_inject(manager: MemoryManager) -> MemoryRecord:
    record = manager.search_memory_records(
        "How should provider timeout retries work?",
        source="collector",
    )[0]
    manager.record_injected_records((record,))
    return record


def test_injected_memory_without_explicit_reference_is_not_adopted(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="fixture", layer="project")

    _retrieve_and_inject(manager)

    assert manager.adopt_injected_records(source="runtime:test") == 0
    assert manager.record_session_outcome("success", source="runtime:test") == 1

    record = manager.read_memory_records(layer="project")[0]
    assert record.retrieval_count == 1
    assert record.injection_count == 1
    assert record.reference_count == 0
    assert record.adoption_count == 0
    assert record.success_count == 0
    assert record.failure_count == 0
    assert record.utility == 0.0
    assert record.last_outcome == "unobserved"


def test_explicit_reference_allows_adoption_and_outcome_feedback(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="fixture", layer="project")

    injected = _retrieve_and_inject(manager)
    assert manager.record_explicit_references(
        f"I relied on memory {injected.memory_id} for the retry policy."
    ) == 1
    assert manager.adopt_injected_records(source="runtime:test") == 1
    assert manager.record_session_outcome("success", source="runtime:test") == 1

    record = manager.read_memory_records(layer="project")[0]
    assert record.retrieval_count == 1
    assert record.injection_count == 1
    assert record.reference_count == 1
    assert record.adoption_count == 1
    assert record.success_count == 1
    assert record.failure_count == 0
    assert record.utility == 1.0
    assert record.last_outcome == "success"
