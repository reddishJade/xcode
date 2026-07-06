"""Tests for read-only governed-memory integrity checks."""

from __future__ import annotations

from pathlib import Path

from xcode.harness.memory import MemoryManager
from xcode.harness.memory.integrity import audit_memory_integrity


def _block() -> str:
    return (
        "## Provider retry policy\n"
        "- Context/Query: Provider timeout retry handling\n"
        "- Solution: Retry transient provider failures with bounded backoff.\n"
        "- Files: src/xcode/provider.py\n"
        "- Takeaways: Preserve the original failure reason.\n"
    )


def test_governed_memory_passes_integrity_audit(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="repl", layer="project")

    report = audit_memory_integrity(manager)

    assert report.ok
    assert report.checked_records == 1
    assert report.governed_records == 1
    assert report.legacy_records == 0
    assert report.issues == ()


def test_legacy_memory_is_visible_but_not_reported_as_governed_failure(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="fixture", layer="project")

    report = audit_memory_integrity(manager)

    assert report.ok
    assert report.governed_records == 0
    assert report.legacy_records == 1


def test_missing_ledger_proposal_is_reported(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="repl", layer="project")
    memory_file = manager.memory_file
    tampered = memory_file.read_text(encoding="utf-8").replace(
        "- Proposal-ID: mp_",
        "- Proposal-ID: mp_missing_",
        1,
    )
    memory_file.write_text(tampered, encoding="utf-8")

    report = audit_memory_integrity(manager)

    assert not report.ok
    assert report.issues[0].code == "missing_proposal"


def test_evidence_mismatch_is_reported(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="repl", layer="project")
    memory_file = manager.memory_file
    tampered = memory_file.read_text(encoding="utf-8").replace(
        "- Ledger-Evidence-IDs: ev_",
        "- Ledger-Evidence-IDs: ev_tampered_",
        1,
    )
    memory_file.write_text(tampered, encoding="utf-8")

    report = audit_memory_integrity(manager)

    assert not report.ok
    assert report.issues[0].code == "evidence_mismatch"
