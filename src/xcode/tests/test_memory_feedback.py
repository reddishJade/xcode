"""Memory 使用归因与结果反馈测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from xcode.harness.memory import MemoryManager


def _block(title: str, solution: str) -> str:
    return (
        f"## {title}\n"
        "- Context/Query: provider timeout handling\n"
        f"- Solution: {solution}\n"
        "- Files: src/provider.py\n"
        "- Takeaways: Attribute feedback only after confirmed use.\n"
    )


def _manager(tmp_path: Path, *, same_user_title: bool = False) -> MemoryManager:
    (tmp_path / "MEMORY.md").write_text(
        _block("Shared title", "PROJECT-SOLUTION"), encoding="utf-8"
    )
    user_file = tmp_path / "user" / "MEMORY.md"
    user_file.parent.mkdir()
    user_file.write_text(
        _block("Shared title" if same_user_title else "User title", "USER-SOLUTION"),
        encoding="utf-8",
    )
    return MemoryManager(
        tmp_path,
        user_memory_file=user_file,
        min_retrieval_score=0.0,
    )


def test_injected_without_attribution_only_updates_exposure(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    record = manager.search_memory_records("provider timeout", limit=1)[0]
    manager.record_injected_records([record])

    manager.record_session_outcome("success")
    updated = manager.read_memory_records(layer=record.layer)[0]

    assert updated.retrieval_count == 1
    assert updated.injection_count == 1
    assert updated.reference_count == 0
    assert updated.adoption_count == 0
    assert updated.success_count == 0
    assert updated.failure_count == 0
    assert updated.utility == pytest.approx(0.0)
    assert updated.last_outcome is None


def test_referenced_record_gets_positive_feedback(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    record = manager.search_memory_records("provider timeout", limit=1)[0]
    manager.record_injected_records([record])

    assert manager.record_explicit_references(record.memory_id) == 1
    manager.record_session_outcome("success")
    updated = manager.read_memory_records(layer=record.layer)[0]

    assert updated.reference_count == 1
    assert updated.success_count == 1
    assert updated.utility == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("outcome", "counter", "utility", "validity"),
    [
        ("failure", "failure_count", -1.0, "needs_review"),
        ("corrected", "correction_count", -0.5, "corrected"),
    ],
)
def test_adopted_record_gets_negative_feedback_and_needs_review(
    tmp_path: Path,
    outcome: Literal["failure", "corrected"],
    counter: str,
    utility: float,
    validity: str,
) -> None:
    manager = _manager(tmp_path)
    record = manager.search_memory_records("provider timeout", limit=1)[0]
    manager.record_adopted_records([record])

    manager.record_session_outcome(outcome)
    updated = manager.read_memory_records(layer=record.layer)[0]

    assert getattr(updated, counter) == 1
    assert updated.utility == pytest.approx(utility)
    assert updated.status == "needs_review"
    assert updated.validity == validity


def test_same_title_across_layers_requires_unambiguous_memory_id(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, same_user_title=True)
    records = manager.search_memory_records("provider timeout", limit=10)
    manager.record_injected_records(records)

    assert manager.record_explicit_references("Shared title") == 0
    project = next(record for record in records if record.layer == "project")
    assert manager.record_explicit_references(project.memory_id) == 1
    manager.record_session_outcome("success")

    project_updated = manager.read_memory_records(layer="project")[0]
    user_updated = manager.read_memory_records(layer="user")[0]
    assert project_updated.success_count == 1
    assert user_updated.success_count == 0
    assert project_updated.reference_count == 1
    assert user_updated.reference_count == 0
