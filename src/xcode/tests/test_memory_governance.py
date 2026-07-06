"""Tests for the evidence ledger and durable-memory promotion gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

from xcode.agent.context_assembly import ContextTrust
from xcode.harness.memory import (
    MemoryEvidenceInput,
    MemoryGovernance,
    MemoryManager,
    MemoryProposalStatus,
)


def _memory_block(title: str = "Verified command") -> str:
    return (
        f"## {title}\n"
        "- Context/Query: Run the project verification command after changing context code.\n"
        "- Solution: Run pytest for focused context tests before the full suite.\n"
        "- Files: src/xcode/agent/context_assembly.py\n"
        "- Takeaways: Preserve the authority boundary while validating behavior.\n"
    )


def test_explicit_user_memory_is_evidence_backed_and_applied() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance = MemoryGovernance(root)

        result = governance.add_explicit_user_memory(
            block=_memory_block(),
            title="Verified command",
            layer="project",
        )

        assert result.decision.reason == "explicit_user_request"
        assert result.proposal.status is MemoryProposalStatus.APPLIED
        assert result.proposal.applied_memory_id
        assert result.proposal.evidence[0].kind == "user_request"
        assert result.proposal.evidence[0].trust is ContextTrust.TRUSTED_USER
        assert governance.ledger.path.exists()

        records = governance.manager.read_memory_records(layer="project")
        assert len(records) == 1
        assert records[0].title == "Verified command"
        assert records[0].validity == "user_asserted"
        assert records[0].evidence[0].kind == "user_request"


def test_untrusted_evidence_is_rejected_before_memory_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance = MemoryGovernance(root)

        result = governance.propose(
            block=_memory_block("Unsafe source"),
            title="Unsafe source",
            layer="project",
            scope=str(root.resolve()),
            source="agent",
            requester="agent",
            evidence=(
                MemoryEvidenceInput(
                    kind="web",
                    reference="search-result:untrusted",
                    trust=ContextTrust.EXTERNAL_UNTRUSTED,
                ),
            ),
        )

        assert result.decision.reason == "untrusted_evidence"
        assert result.proposal.status is MemoryProposalStatus.REJECTED
        assert governance.manager.read_memory_records(layer="project") == []


def test_automation_remains_pending_until_explicit_approval() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance = MemoryGovernance(root)

        result = governance.propose(
            block=_memory_block("Verified automation"),
            title="Verified automation",
            layer="project",
            scope=str(root.resolve()),
            source="validation",
            requester="automation",
            evidence=(
                MemoryEvidenceInput(
                    kind="tool_result",
                    reference="tool:pytest:exit=0",
                    trust=ContextTrust.VERIFIED_TOOL,
                    scope=str(root.resolve()),
                ),
            ),
        )

        assert result.decision.requires_approval
        assert result.proposal.status is MemoryProposalStatus.PENDING
        assert governance.manager.read_memory_records(layer="project") == []

        applied = governance.approve(result.proposal.proposal_id)

        assert applied.status is MemoryProposalStatus.APPLIED
        records = governance.manager.read_memory_records(layer="project")
        assert len(records) == 1
        assert records[0].validity == "verified"
        assert records[0].evidence[0].reference == "tool:pytest:exit=0"


def test_missing_evidence_is_rejected_before_manager_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance = MemoryGovernance(root)

        result = governance.propose(
            block=_memory_block("No evidence"),
            title="No evidence",
            layer="project",
            scope=str(root.resolve()),
            source="agent",
            requester="agent",
            evidence=(),
        )

        assert result.proposal.status is MemoryProposalStatus.REJECTED
        assert result.decision.reason == "missing_evidence"
        assert governance.manager.read_memory_records(layer="project") == []


def test_public_manager_routes_repl_write_through_governance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = MemoryManager(root)

        assert manager.add_memory_block(
            _memory_block("REPL memory"),
            source="repl",
            layer="project",
        )

        proposals = manager.governance.ledger.list_proposals()
        assert len(proposals) == 1
        assert proposals[0].status is MemoryProposalStatus.APPLIED
        assert proposals[0].source == "repl"
        assert proposals[0].evidence[0].kind == "user_request"
        assert manager.read_memory_records(layer="project")[0].title == "REPL memory"


def test_compaction_creates_pending_proposal_without_writing_memory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = MemoryManager(root)
        summary = _memory_block("Compaction candidate")

        manager.consolidate(summary)

        proposals = manager.governance.ledger.list_proposals()
        assert len(proposals) == 1
        assert proposals[0].status is MemoryProposalStatus.PENDING
        assert proposals[0].source == "consolidation"
        assert proposals[0].evidence[0].kind == "compaction_summary"
        assert manager.read_memory_records(layer="project") == []


def test_structured_compaction_creates_pending_decision_proposal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = MemoryManager(root)
        summary = (
            "[Compressed]\n"
            "## Key Decisions\n"
            "- Context authority: workspace instructions must not become system policy\n"
        )

        manager.consolidate_structured(summary)

        proposals = manager.governance.ledger.list_proposals()
        assert len(proposals) == 1
        assert proposals[0].status is MemoryProposalStatus.PENDING
        assert proposals[0].title.startswith("Decision:")
        assert manager.read_memory_records(layer="project") == []
