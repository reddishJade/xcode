"""长期记忆维护、归档和原子写入测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xcode.harness.memory import MemoryManager
from xcode.harness.memory import MemoryRetrievalContext


def _block(
    title: str,
    solution: str,
    *,
    status: str = "active",
    validity: str = "derived",
    success_count: int = 0,
    evidence: str = "",
) -> str:
    lines = [
        f"## {title}",
        "- Context/Query: 维护 provider timeout 规则",
        f"- Solution: {solution}",
        "- Files: src/provider.py",
        "- Takeaways: 中英文记录都必须保持可维护和可检索。",
        f"- Status: {status}",
        f"- Validity: {validity}",
        f"- Success-Count: {success_count}",
    ]
    if evidence:
        lines.append(f"- Evidence: {evidence}")
    return "\n".join(lines) + "\n"


def _manager(tmp_path: Path, blocks: tuple[str, ...]) -> MemoryManager:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "MEMORY.md").write_text("\n".join(blocks), encoding="utf-8")
    user_file = tmp_path / "user" / "MEMORY.md"
    user_file.parent.mkdir(exist_ok=True)
    user_file.write_text("", encoding="utf-8")
    return MemoryManager(tmp_path, user_memory_file=user_file)


def test_maintenance_dry_run_is_strictly_read_only(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        (
            _block("重复一", "重试一次。", evidence="test:first"),
            _block("重复二", "重试一次。", evidence="commit:second"),
            _block("候选", "限制退避。", status="candidate", success_count=2),
            _block("待审", "不要重试。", status="needs_review"),
        ),
    )

    before = manager.memory_file.read_bytes()
    report = manager.maintain_memory()
    assert not report.applied
    assert report.duplicate_merges
    assert report.candidate_promotions
    assert report.needs_review
    assert manager.memory_file.read_bytes() == before
    assert not manager.archive_dir.exists()
    assert not manager.lru_file.exists()


def test_maintenance_apply_promotes_merges_and_archives_superseded(
    tmp_path: Path,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block("Duplicate A", "Retry once.", evidence="test:first"),
            _block("Duplicate B", "Retry once.", evidence="commit:second"),
            _block("Candidate", "Bound backoff.", status="candidate", success_count=2),
            _block("Superseded", "Retry forever.", status="superseded"),
        ),
    )
    report = manager.maintain_memory(apply=True)
    assert report.applied
    records = manager.read_memory_records(layer="project")
    assert len(records) == 2
    duplicate = next(record for record in records if record.evidence)
    assert {item.kind for item in duplicate.evidence} == {"test", "commit"}
    candidate = next(record for record in records if record.title == "Candidate")
    assert (candidate.status, candidate.validity) == ("active", "verified")
    archives = list(manager.archive_dir.glob("*.md"))
    assert len(archives) == 2
    assert any("Retry forever" in path.read_text(encoding="utf-8") for path in archives)


def test_maintenance_merge_preserves_contradiction_quarantine(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        (
            _block(
                "Verified duplicate",
                "Retry once.",
                status="active",
                validity="verified",
            ),
            _block(
                "Contradicted duplicate",
                "Retry once.",
                status="needs_review",
                validity="contradicted",
            ),
        ),
    )

    report = manager.maintain_memory(apply=True)

    assert report.applied
    records = manager.read_memory_records(layer="project")
    assert len(records) == 1
    assert records[0].status == "needs_review"
    assert records[0].validity == "contradicted"
    assert (
        manager.search_memory_records(
            "provider timeout",
            source="prompt",
            retrieval_context=MemoryRetrievalContext(query="provider timeout"),
        )
        == []
    )


@pytest.mark.parametrize("terminal_status", ["superseded", "obsolete", "deprecated"])
def test_terminal_and_active_duplicate_merge_is_order_independent(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    active = _block(
        "Active duplicate",
        "Retry once.",
        status="active",
        validity="verified",
        evidence="test:active-proof",
    )
    terminal = _block(
        "Terminal duplicate",
        "Retry once.",
        status=terminal_status,
        validity="derived",
        evidence=f"commit:{terminal_status}-proof",
    )
    outcomes: list[tuple[str, str, str, str, tuple[tuple[str, str], ...]]] = []

    for reversed_order, blocks in enumerate(((terminal, active), (active, terminal))):
        manager = _manager(tmp_path / str(reversed_order), blocks)
        report = manager.maintain_memory(apply=True)

        assert report.applied
        records = manager.read_memory_records(layer="project")
        assert len(records) == 1
        record = records[0]
        assert (record.status, record.validity) == ("needs_review", "needs_review")
        assert {(item.kind, item.reference) for item in record.evidence} == {
            ("test", "active-proof"),
            ("commit", f"{terminal_status}-proof"),
        }
        assert len(list(manager.archive_dir.glob("*.md"))) == 1
        assert (
            manager.search_memory_records(
                "provider timeout",
                source="prompt",
                retrieval_context=MemoryRetrievalContext(query="provider timeout"),
            )
            == []
        )
        outcomes.append(
            (
                record.title,
                record.memory_id,
                record.status,
                record.validity,
                tuple(sorted((item.kind, item.reference) for item in record.evidence)),
            )
        )

    assert outcomes[0] == outcomes[1]


def test_maintenance_refuses_to_overwrite_changed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, (_block("Candidate", "Retry once."),))
    original = manager._maintenance_duplicate_groups
    changed = False

    def mutate(records: object) -> object:
        nonlocal changed
        if not changed:
            changed = True
            manager.memory_file.write_text(
                manager.memory_file.read_text(encoding="utf-8")
                + "\n<!-- user edit -->\n",
                encoding="utf-8",
            )
        return original(records)  # type: ignore[arg-type]

    monkeypatch.setattr(manager, "_maintenance_duplicate_groups", mutate)
    report = manager.maintain_memory(apply=True)
    assert not report.applied
    assert report.conflicts == ("project",)
    assert "user edit" in manager.memory_file.read_text(encoding="utf-8")


def test_atomic_write_failure_preserves_original_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, (_block("Original", "Retry once."),))
    before = manager.memory_file.read_bytes()

    def fail_replace(_source: os.PathLike[str], _target: os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        manager._write_blocks([_block("Changed", "Never retry.")], "project")
    assert manager.memory_file.read_bytes() == before
