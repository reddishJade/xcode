"""Layer-aware governance for project and user durable memory.

A user-global MEMORY.md must not depend on the repository where it was first
created. This coordinator keeps project proposals in the project ledger while
storing user-layer proposals alongside the user memory root, so provenance stays
explainable after switching repositories.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence, cast

from xcode.agent.context_assembly import ContextTrust

from .governance import (
    MemoryEvidenceInput,
    MemoryGovernance,
    MemoryLedger,
    MemoryOperation,
    MemoryPromotionPolicy,
    MemoryProposal,
    MemoryProposalResult,
    MemoryProposalStatus,
    MemoryRequester,
    _memory_id_for_title,
    _validity_for_evidence,
)
from .manager import MemoryLayer, MemoryManager
from .parsing import MemoryRecord, MemoryType, parse_fields, parse_memory_record
from .retirement import retire_memory_record
from .retirement_policy import RetirementPromotionPolicy


class LayeredMemoryGovernance(MemoryGovernance):
    """Apply one promotion contract across layer-specific governance ledgers."""

    def __init__(
        self,
        project_root: Path,
        manager: MemoryManager,
        policy: MemoryPromotionPolicy | None = None,
    ) -> None:
        super().__init__(
            project_root,
            manager=manager,
            policy=policy or RetirementPromotionPolicy(),
        )
        self._user_ledger = MemoryLedger(_user_ledger_root(manager))

    def ledger_for(self, layer: MemoryLayer) -> MemoryLedger:
        """Return the ledger whose lifetime matches a durable memory layer."""
        return self.ledger if layer == "project" else self._user_ledger

    def ledgers(self) -> tuple[MemoryLedger, ...]:
        """Return unique ledgers for inspection tooling."""
        if self.ledger.path == self._user_ledger.path:
            return (self.ledger,)
        return (self.ledger, self._user_ledger)

    def propose(
        self,
        *,
        block: str,
        title: str,
        layer: MemoryLayer,
        scope: str,
        source: str,
        requester: MemoryRequester,
        evidence: Sequence[MemoryEvidenceInput],
        memory_type: MemoryType | None = None,
        operation: MemoryOperation = "add",
    ) -> MemoryProposalResult:
        ledger = self.ledger_for(layer)
        recorded_evidence = ledger.record_evidence(evidence)
        proposal = ledger.create_proposal(
            operation=operation,
            title=title,
            block=block,
            layer=layer,
            scope=scope,
            source=source,
            requester=requester,
            memory_type=memory_type,
            evidence=recorded_evidence,
        )
        decision = self.policy.decide(
            operation=operation,
            layer=layer,
            requester=requester,
            evidence=recorded_evidence,
        )
        if decision.status is MemoryProposalStatus.REJECTED:
            proposal = ledger.resolve(
                proposal.proposal_id,
                status=MemoryProposalStatus.REJECTED,
                reason=decision.reason,
            )
            return MemoryProposalResult(proposal=proposal, decision=decision)
        if decision.status is MemoryProposalStatus.APPROVED:
            ledger.resolve(
                proposal.proposal_id,
                status=MemoryProposalStatus.APPROVED,
                reason=decision.reason,
            )
            return MemoryProposalResult(
                proposal=self.apply(proposal.proposal_id),
                decision=decision,
            )
        return MemoryProposalResult(proposal=proposal, decision=decision)

    def retire_explicit_user_memory(
        self,
        record: MemoryRecord,
        *,
        source: str = "cli",
        reason: str = "retired_by_user",
    ) -> MemoryProposalResult:
        """Create and apply an evidence-backed retirement from direct user intent."""
        layer = cast(MemoryLayer, record.layer)
        scope = record.scope or (
            "user_global" if layer == "user" else str(self.project_root)
        )
        original_hash = sha256(record.block.encode("utf-8")).hexdigest()
        proposal_block = record.block.rstrip() + f"\n- Retirement-Reason: {reason}\n"
        return self.propose(
            block=proposal_block,
            title=record.title,
            layer=layer,
            scope=scope,
            source=source,
            requester="explicit_user",
            operation="retire",
            evidence=(
                MemoryEvidenceInput(
                    kind="user_request",
                    reference=f"{source}:memory-retire:{original_hash[:16]}",
                    trust=ContextTrust.TRUSTED_USER,
                    scope=scope,
                    content_hash=original_hash,
                ),
            ),
        )

    def approve(self, proposal_id: str, *, approver: str = "user") -> MemoryProposal:
        ledger, proposal = self._proposal_location(proposal_id)
        if proposal.status is not MemoryProposalStatus.PENDING:
            raise ValueError(f"proposal is not pending: {proposal.status}")
        ledger.resolve(
            proposal_id,
            status=MemoryProposalStatus.APPROVED,
            reason=f"approved_by:{approver}",
        )
        return self.apply(proposal_id)

    def reject(self, proposal_id: str, *, reason: str = "rejected_by_user") -> MemoryProposal:
        ledger, proposal = self._proposal_location(proposal_id)
        if proposal.status is not MemoryProposalStatus.PENDING:
            raise ValueError(f"proposal is not pending: {proposal.status}")
        return ledger.resolve(
            proposal_id,
            status=MemoryProposalStatus.REJECTED,
            reason=reason,
        )

    def apply(self, proposal_id: str) -> MemoryProposal:
        ledger, proposal = self._proposal_location(proposal_id)
        if proposal.status is not MemoryProposalStatus.APPROVED:
            raise ValueError(f"proposal is not approved: {proposal.status}")

        if proposal.operation == "retire":
            target = parse_memory_record(proposal.block, layer=proposal.layer)
            reason = parse_fields(proposal.block).get(
                "retirement-reason",
                "retired_by_governance",
            )
            retired = retire_memory_record(
                self.manager,
                target.memory_id,
                layer=proposal.layer,
                reason=reason,
            )
            if not retired:
                return ledger.resolve(
                    proposal_id,
                    status=MemoryProposalStatus.FAILED,
                    reason="retirement_target_missing",
                )
            return ledger.resolve(
                proposal_id,
                status=MemoryProposalStatus.APPLIED,
                reason="retired_memory_record",
                applied_memory_id=target.memory_id,
            )

        if proposal.operation != "add":
            return ledger.resolve(
                proposal_id,
                status=MemoryProposalStatus.FAILED,
                reason="operation_not_implemented",
            )

        persisted = self.manager.add_memory_block(
            proposal.block,
            source=f"governance:{proposal.source}",
            scope=proposal.scope,
            memory_type=proposal.memory_type,
            status="active",
            validity=_validity_for_evidence(proposal.evidence),
            evidence=tuple(item.to_memory_evidence() for item in proposal.evidence),
            layer=proposal.layer,
        )
        if not persisted:
            return ledger.resolve(
                proposal_id,
                status=MemoryProposalStatus.FAILED,
                reason="memory_manager_rejected_candidate",
            )
        return ledger.resolve(
            proposal_id,
            status=MemoryProposalStatus.APPLIED,
            reason="applied_to_memory_manager",
            applied_memory_id=_memory_id_for_title(
                self.manager,
                proposal.layer,
                proposal.title,
            ),
        )

    def get_proposal(self, proposal_id: str) -> MemoryProposal:
        """Read a proposal regardless of the layer where it originated."""
        _, proposal = self._proposal_location(proposal_id)
        return proposal

    def _proposal_location(self, proposal_id: str) -> tuple[MemoryLedger, MemoryProposal]:
        for ledger in self.ledgers():
            try:
                return ledger, ledger.get_proposal(proposal_id)
            except KeyError:
                continue
        raise KeyError(f"unknown memory proposal: {proposal_id}")


def _user_ledger_root(manager: MemoryManager) -> Path:
    """Choose a stable user-local root independent of the active repository."""
    return manager.user_memory_file.parent
