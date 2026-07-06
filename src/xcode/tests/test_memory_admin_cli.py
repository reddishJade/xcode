"""Tests for user-operated durable-memory administration commands."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from xcode.cli.memory_cmd import handle_memory_command
from xcode.main import parse_args
from xcode.harness.memory import MemoryManager, MemoryProposalStatus


def _block() -> str:
    return (
        "## Compaction candidate\n"
        "- Context/Query: Context trust boundaries\n"
        "- Solution: Keep workspace instructions outside system authority.\n"
        "- Files: src/xcode/agent/context_assembly.py\n"
        "- Takeaways: Automation-derived memory requires review.\n"
    )


def _approve_args(proposal_id: str, *, yes: bool) -> Namespace:
    return Namespace(
        memory_action="approve",
        proposal_id=proposal_id,
        yes=yes,
        approver="test_user",
        status="pending",
    )


def test_memory_approve_requires_yes_and_then_applies(tmp_path: Path, capsys) -> None:
    manager = MemoryManager(tmp_path)
    manager.consolidate(_block())
    proposal = manager.governance.ledger_for("project").list_proposals()[0]

    code = handle_memory_command(_approve_args(proposal.proposal_id, yes=False), tmp_path)

    assert code == 2
    assert "Re-run with --yes" in capsys.readouterr().out
    assert manager.governance.get_proposal(proposal.proposal_id).status is MemoryProposalStatus.PENDING
    assert manager.read_memory_records(layer="project") == []

    code = handle_memory_command(_approve_args(proposal.proposal_id, yes=True), tmp_path)

    assert code == 0
    assert "Approved and applied" in capsys.readouterr().out
    assert manager.governance.get_proposal(proposal.proposal_id).status is MemoryProposalStatus.APPLIED
    assert len(manager.read_memory_records(layer="project")) == 1


def test_memory_reject_requires_yes_and_never_writes(tmp_path: Path, capsys) -> None:
    manager = MemoryManager(tmp_path)
    manager.consolidate(_block())
    proposal = manager.governance.ledger_for("project").list_proposals()[0]
    args = Namespace(
        memory_action="reject",
        proposal_id=proposal.proposal_id,
        yes=False,
        reason="not_reusable",
        status="pending",
    )

    code = handle_memory_command(args, tmp_path)

    assert code == 2
    assert "Re-run with --yes" in capsys.readouterr().out
    assert manager.governance.get_proposal(proposal.proposal_id).status is MemoryProposalStatus.PENDING

    args.yes = True
    code = handle_memory_command(args, tmp_path)

    assert code == 0
    assert "Rejected" in capsys.readouterr().out
    assert manager.governance.get_proposal(proposal.proposal_id).status is MemoryProposalStatus.REJECTED
    assert manager.read_memory_records(layer="project") == []


def test_memory_audit_needs_no_provider_configuration(tmp_path: Path, capsys) -> None:
    args = Namespace(memory_action="audit", status="pending")

    code = handle_memory_command(args, tmp_path)

    assert code == 0
    assert "Memory integrity:" in capsys.readouterr().out


def test_memory_parser_defaults_to_pending_proposals(tmp_path: Path) -> None:
    args = parse_args(["--project-root", str(tmp_path), "memory"])

    assert args.command == "memory"
    assert args.memory_action == "proposals"
    assert args.status == "pending"
