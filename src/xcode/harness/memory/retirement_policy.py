"""Promotion policy extension for governed memory retirement."""

from __future__ import annotations

from typing import Sequence

from xcode.agent.context_assembly import ContextTrust

from .governance import (
    GovernedMemoryEvidence,
    MemoryOperation,
    MemoryPromotionDecision,
    MemoryPromotionPolicy,
    MemoryProposalStatus,
    MemoryRequester,
)
from .manager import MemoryLayer


class RetirementPromotionPolicy(MemoryPromotionPolicy):
    """Allow explicit-user retirement while keeping automation reviewable."""

    def decide(
        self,
        *,
        operation: MemoryOperation,
        layer: MemoryLayer,
        requester: MemoryRequester,
        evidence: Sequence[GovernedMemoryEvidence],
    ) -> MemoryPromotionDecision:
        if operation != "retire":
            return super().decide(
                operation=operation,
                layer=layer,
                requester=requester,
                evidence=evidence,
            )
        if not evidence:
            return MemoryPromotionDecision(
                MemoryProposalStatus.REJECTED,
                "missing_evidence",
            )
        if any(
            item.trust
            in {ContextTrust.EXTERNAL_UNTRUSTED, ContextTrust.WORKSPACE_UNTRUSTED}
            for item in evidence
        ):
            return MemoryPromotionDecision(
                MemoryProposalStatus.REJECTED,
                "untrusted_evidence",
            )
        if requester == "explicit_user":
            return MemoryPromotionDecision(
                MemoryProposalStatus.APPROVED,
                "explicit_user_retirement",
            )
        return MemoryPromotionDecision(
            MemoryProposalStatus.PENDING,
            "retirement_requires_promotion_approval",
        )
