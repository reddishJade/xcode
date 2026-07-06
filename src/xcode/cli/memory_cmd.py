"""User-operated durable-memory proposal administration.

This module deliberately does not expose approval as an agent tool. It is invoked
by the local user through ``xcode memory`` and requires ``--yes`` for every
state-changing operation.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from xcode.harness.memory import (
    MemoryManager,
    MemoryProposal,
    MemoryProposalStatus,
    audit_memory_integrity,
)


def handle_memory_command(args: Namespace, project_root: Path) -> int:
    """Execute a user-owned memory administration subcommand."""
    manager = MemoryManager(project_root)
    action = str(args.memory_action or "proposals")

    if action == "proposals":
        return _list_proposals(manager, str(args.status))
    if action == "approve":
        return _approve(
            manager,
            str(args.proposal_id),
            yes=bool(args.yes),
            approver=str(args.approver),
        )
    if action == "reject":
        return _reject(
            manager,
            str(args.proposal_id),
            yes=bool(args.yes),
            reason=str(args.reason),
        )
    if action == "audit":
        return _audit(manager)

    print(f"Unknown memory action: {action}")
    return 2


def _list_proposals(manager: MemoryManager, raw_status: str) -> int:
    status = raw_status.strip().lower() or "pending"
    valid = {"all", *(item.value for item in MemoryProposalStatus)}
    if status not in valid:
        print("status must be one of: " + ", ".join(sorted(valid)))
        return 2
    proposals = _all_proposals(manager)
    if status != "all":
        proposals = tuple(
            proposal for proposal in proposals if proposal.status.value == status
        )
    if not proposals:
        print(f"No {status if status != 'all' else 'matching'} memory proposals.")
        return 0

    print(f"Memory proposals ({len(proposals)}):")
    for proposal in proposals:
        print(
            f"- [{proposal.status.value}] {proposal.proposal_id}: {proposal.title} "
            f"(layer={proposal.layer}, source={proposal.source})"
        )
        print(f"  scope={proposal.scope}")
        if proposal.decision_reason:
            print(f"  decision={proposal.decision_reason}")
        for evidence in proposal.evidence:
            print(
                f"  evidence={evidence.evidence_id} {evidence.kind}:"
                f"{evidence.reference} trust={evidence.trust.value}"
            )
    return 0


def _approve(
    manager: MemoryManager,
    proposal_id: str,
    *,
    yes: bool,
    approver: str,
) -> int:
    proposal = _get_proposal(manager, proposal_id)
    if proposal is None:
        print(f"Unknown memory proposal: {proposal_id}")
        return 2
    if proposal.status is not MemoryProposalStatus.PENDING:
        print(f"Proposal {proposal_id} is not pending: {proposal.status.value}")
        return 2
    _print_confirmation_preview("approve", proposal)
    if not yes:
        print("Re-run with --yes to approve and write this memory.")
        return 2

    applied = manager.governance.approve(proposal_id, approver=approver)
    print(
        f"Approved and applied {applied.proposal_id}: "
        f"memory_id={applied.applied_memory_id or '(unknown)'}"
    )
    return 0


def _reject(
    manager: MemoryManager,
    proposal_id: str,
    *,
    yes: bool,
    reason: str,
) -> int:
    proposal = _get_proposal(manager, proposal_id)
    if proposal is None:
        print(f"Unknown memory proposal: {proposal_id}")
        return 2
    if proposal.status is not MemoryProposalStatus.PENDING:
        print(f"Proposal {proposal_id} is not pending: {proposal.status.value}")
        return 2
    _print_confirmation_preview("reject", proposal)
    if not yes:
        print("Re-run with --yes to reject this memory proposal.")
        return 2

    rejected = manager.governance.reject(proposal_id, reason=reason)
    print(f"Rejected {rejected.proposal_id}: {rejected.decision_reason}")
    return 0


def _audit(manager: MemoryManager) -> int:
    report = audit_memory_integrity(manager)
    print(
        "Memory integrity: "
        f"checked={report.checked_records} governed={report.governed_records} "
        f"legacy={report.legacy_records} status={'ok' if report.ok else 'failed'}"
    )
    for issue in report.issues:
        print(
            f"- {issue.code} memory={issue.memory_id} "
            f"layer={issue.layer}: {issue.detail}"
        )
    return 0 if report.ok else 1


def _all_proposals(manager: MemoryManager) -> tuple[MemoryProposal, ...]:
    proposals = [
        proposal
        for ledger in manager.governance.ledgers()
        for proposal in ledger.list_proposals()
    ]
    return tuple(
        sorted(
            proposals,
            key=lambda proposal: (proposal.created_at, proposal.proposal_id),
        )
    )


def _get_proposal(manager: MemoryManager, proposal_id: str) -> MemoryProposal | None:
    try:
        return manager.governance.get_proposal(proposal_id)
    except KeyError:
        return None


def _print_confirmation_preview(action: str, proposal: MemoryProposal) -> None:
    print(f"About to {action} memory proposal {proposal.proposal_id}:")
    print(f"  title={proposal.title}")
    print(f"  layer={proposal.layer} scope={proposal.scope}")
    print(f"  source={proposal.source} requester={proposal.requester}")
    for evidence in proposal.evidence:
        print(
            f"  evidence={evidence.evidence_id} {evidence.kind}:"
            f"{evidence.reference} trust={evidence.trust.value}"
        )
