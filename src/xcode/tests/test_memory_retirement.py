"""Tests for evidence-backed, reversible durable-memory retirement."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from xcode.agent.context_assembly import ContextTrust
from xcode.cli.memory_cmd import handle_memory_command
from xcode.harness.memory import (
    MemoryEvidenceInput,
    MemoryManager,
    MemoryProposalStatus,
)


def _block() -> str:
    return (
        "## Old provider retry policy\n"
        "- Context/Query: Provider timeout retry handling\n"
        "- Solution: Retry every failure forever.\n"
        "- Files: src/xcode/provider.py\n"
        "- Takeaways: Legacy retry behavior that must be retired.\n"
    )


def test_explicit_retirement_removes_active_record_and_archives_evidence(
    tmp_path: Path,
) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="fixture", layer="project")
    record = manager.read_memory_records(layer="project")[0]
    manager.drain_trace_events()

    result = manager.governance.retire_explicit_user_memory(
        record,
        source="test",
        reason="superseded_by_bounded_retry",
    )

    assert result.proposal.operation == "retire"
    assert result.proposal.status is MemoryProposalStatus.APPLIED
    assert result.proposal.applied_memory_id == record.memory_id
    assert manager.read_memory_records(layer="project") == []
    archived = list(manager.archive_dir.glob("retired_*.md"))
    assert len(archived) == 1
    archived_text = archived[0].read_text(encoding="utf-8")
    assert record.memory_id in archived_text
    assert "Retirement-Reason: superseded_by_bounded_retry" in archived_text
    assert "Retired-At:" in archived_text
    assert result.proposal.evidence[0].trust is ContextTrust.TRUSTED_USER
    assert result.proposal.evidence[0].kind == "user_request"
    events = manager.drain_trace_events()
    assert [event.type for event in events] == ["forgotten"]
    assert events[0].source == "retirement"


def test_automated_retirement_is_pending_until_human_approval(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="fixture", layer="project")
    record = manager.read_memory_records(layer="project")[0]

    result = manager.governance.propose(
        block=record.block,
        title=record.title,
        layer="project",
        scope=str(tmp_path.resolve()),
        source="retention_review",
        requester="automation",
        operation="retire",
        evidence=(
            MemoryEvidenceInput(
                kind="tool_result",
                reference="tool:validation:obsolete-policy",
                trust=ContextTrust.VERIFIED_TOOL,
            ),
        ),
    )

    assert result.proposal.status is MemoryProposalStatus.PENDING
    assert manager.read_memory_records(layer="project")[0].memory_id == record.memory_id


def test_cli_retirement_requires_yes_before_creating_retirement_proposal(
    tmp_path: Path,
    capsys,
) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_block(), source="fixture", layer="project")
    record = manager.read_memory_records(layer="project")[0]
    args = Namespace(
        memory_action="retire",
        memory_id=record.memory_id,
        yes=False,
        reason="obsolete",
        status="pending",
    )

    code = handle_memory_command(args, tmp_path)

    assert code == 2
    assert "Re-run with --yes" in capsys.readouterr().out
    assert manager.read_memory_records(layer="project")[0].memory_id == record.memory_id
    assert manager.governance.ledger_for("project").list_proposals() == ()

    args.yes = True
    code = handle_memory_command(args, tmp_path)

    assert code == 0
    assert "Retired" in capsys.readouterr().out
    proposal = manager.governance.ledger_for("project").list_proposals()[0]
    assert proposal.operation == "retire"
    assert proposal.status is MemoryProposalStatus.APPLIED
    assert manager.read_memory_records(layer="project") == []
