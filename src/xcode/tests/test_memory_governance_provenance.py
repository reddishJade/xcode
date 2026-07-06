"""Roundtrip tests for governed durable-memory provenance."""

from __future__ import annotations

from pathlib import Path

from xcode.agent.context_collector import ContextCollectionInput
from xcode.agent.messages import UserMessage
from xcode.harness.memory import MemoryCollector, MemoryManager, MemoryProposalStatus


def _block(title: str = "Provider retry policy") -> str:
    return (
        f"## {title}\n"
        "- Context/Query: Provider timeout retry handling\n"
        "- Solution: Retry transient failures with bounded backoff.\n"
        "- Files: src/xcode/provider.py\n"
        "- Takeaways: Preserve the original failure reason.\n"
    )


def test_explicit_user_memory_roundtrips_proposal_and_evidence_ids(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)

    assert manager.add_memory_block(_block(), source="repl", layer="project")

    proposal = manager.governance.ledger.list_proposals()[0]
    record = manager.read_memory_records(layer="project")[0]
    assert proposal.status is MemoryProposalStatus.APPLIED
    assert record.fields["proposal-id"] == proposal.proposal_id
    assert record.fields["ledger-evidence-ids"] == proposal.evidence[0].evidence_id
    assert record.fields["evidence"] == "user_request:repl:memory-add:" + (
        proposal.evidence[0].content_hash[:16]
    )


def test_memory_collector_exposes_only_real_evidence_ids_as_provenance(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="repl", layer="project")
    proposal = manager.governance.ledger.list_proposals()[0]
    collector = MemoryCollector(manager, project_root=tmp_path)

    blocks = collector.collect(
        ContextCollectionInput(
            project_root=tmp_path,
            messages=[UserMessage(content="How should provider timeout retries work?")],
        )
    )

    assert len(blocks) == 1
    provenance = blocks[0].provenance
    assert provenance.evidence_ids == (proposal.evidence[0].evidence_id,)
    assert f"memory:{manager.read_memory_records()[0].memory_id}" in provenance.locator
    assert f"proposal:{proposal.proposal_id}" in provenance.locator
    assert f"proposal={proposal.proposal_id}" in blocks[0].content


def test_approved_consolidation_roundtrips_proposal_and_evidence_ids(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    manager.consolidate(_block("Compaction retry policy"))
    pending = manager.governance.ledger.list_proposals()[0]

    applied = manager.governance.approve(pending.proposal_id)

    assert applied.status is MemoryProposalStatus.APPLIED
    record = manager.read_memory_records(layer="project")[0]
    assert record.fields["proposal-id"] == pending.proposal_id
    assert record.fields["ledger-evidence-ids"] == pending.evidence[0].evidence_id
    assert record.fields["evidence"].startswith("compaction_summary:summary:")
