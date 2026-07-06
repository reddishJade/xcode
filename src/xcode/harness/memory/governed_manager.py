"""Governed public MemoryManager facade.

The underlying :mod:`manager` remains the storage and retrieval engine. This
subclass intercepts only externally meaningful durable-write origins:

* ``source="repl"`` is an explicit user request and therefore creates an
  evidence-backed proposal that may apply immediately.
* compaction consolidation is automation and therefore creates pending proposals
  instead of writing directly to ``MEMORY.md``.
* a governance-approved write receives immutable proposal and evidence IDs before
  it reaches the Markdown store, preserving a complete read-back audit chain.

All other callers retain the base manager's behavior so low-level migration,
fixtures, and deterministic storage operations are not silently reclassified as
user intent.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

from xcode.agent.context_assembly import ContextTrust

from .governance import MemoryEvidenceInput, MemoryProposal, MemoryProposalStatus
from .layered_governance import LayeredMemoryGovernance
from .manager import (
    MemoryLayer,
    MemoryManager as BaseMemoryManager,
    MemoryRerankPolicy,
)
from .parsing import MemoryEvidence, MemoryType, extract_title


class GovernedMemoryManager(BaseMemoryManager):
    """Public manager that routes user and consolidation writes through policy."""

    def __init__(
        self,
        root: Path,
        max_blocks: int = 0,
        user_memory_file: Path | None = None,
        min_retrieval_score: float = 0.2,
        min_confidence: float = 0.0,
        rerank_policy: MemoryRerankPolicy | None = None,
        *,
        governance: LayeredMemoryGovernance | None = None,
    ) -> None:
        super().__init__(
            root,
            max_blocks=max_blocks,
            user_memory_file=user_memory_file,
            min_retrieval_score=min_retrieval_score,
            min_confidence=min_confidence,
            rerank_policy=rerank_policy,
        )
        self._governance = governance

    @property
    def governance(self) -> LayeredMemoryGovernance:
        """Lazily create layer-aware governance bound to this storage manager."""
        if self._governance is None:
            self._governance = LayeredMemoryGovernance(self.root, manager=self)
        return self._governance

    def add_memory_block(
        self,
        block: str,
        *,
        source: str | None = None,
        scope: str | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        status: str | None = None,
        validity: str | None = None,
        supersedes: Sequence[str] = (),
        evidence: Sequence[MemoryEvidence] = (),
        layer: MemoryLayer = "project",
    ) -> bool:
        """Route user intent and approved governance writes through explicit paths."""
        if source == "repl":
            title = extract_title(block)
            if not title:
                return False
            effective_scope = scope or (
                "user_global" if layer == "user" else str(self.root.resolve())
            )
            result = self.governance.add_explicit_user_memory(
                block=block,
                title=title,
                layer=layer,
                scope=effective_scope,
                source="repl",
                memory_type=memory_type,
            )
            return result.proposal.status is MemoryProposalStatus.APPLIED

        if source is not None and source.startswith("governance:"):
            block = self._attach_governance_provenance(block, source=source, layer=layer)

        return super().add_memory_block(
            block,
            source=source,
            scope=scope,
            confidence=confidence,
            memory_type=memory_type,
            status=status,
            validity=validity,
            supersedes=supersedes,
            evidence=evidence,
            layer=layer,
        )

    def consolidate(self, summary: str) -> None:
        """Create pending proposals from legacy compact-summary candidates."""
        for block in self._extract_summary_blocks(summary):
            if self._is_memory_attempt(block):
                self._propose_consolidation_candidate(block, summary)

    def consolidate_structured(self, summary: str) -> None:
        """Create pending proposals from structured compact-summary candidates."""
        sections = self._parse_structured_summary(summary)
        for decision_text in self._extract_bullet_items(sections.get("key decisions", "")):
            block = self._decision_to_memory_block(decision_text)
            if block is not None:
                self._propose_consolidation_candidate(block, summary)

        goal = sections.get("goal", "").strip()
        if goal and len(goal) > 30 and self._should_seed_project_context(goal):
            block = (
                "## Project context\n"
                f"- Context/Query: {goal.split(chr(10))[0] if chr(10) in goal else goal}\n"
                "- Solution: (learn from ongoing work)\n"
                "- Files: (see project)\n"
                f"- Takeaways: {goal[:200]}"
            )
            self._propose_consolidation_candidate(block, summary)

    def _propose_consolidation_candidate(self, block: str, summary: str) -> None:
        """Preserve the old scope filter but never bypass promotion approval."""
        if not self._has_reusable_scope(block):
            self._ingest_consolidation_candidate(
                block,
                source="consolidation",
                layer="project",
            )
            return

        title = extract_title(block)
        if not title:
            return
        summary_hash = sha256(summary.encode("utf-8")).hexdigest()
        self.governance.propose(
            block=block,
            title=title,
            layer="project",
            scope=str(self.root.resolve()),
            source="consolidation",
            requester="consolidation",
            evidence=(
                MemoryEvidenceInput(
                    kind="compaction_summary",
                    reference=f"summary:{summary_hash[:24]}",
                    trust=ContextTrust.RUNTIME_INTERNAL,
                    scope=str(self.root.resolve()),
                    content_hash=summary_hash,
                ),
            ),
        )

    def _attach_governance_provenance(
        self,
        block: str,
        *,
        source: str,
        layer: MemoryLayer,
    ) -> str:
        """Add immutable ledger backreferences to the matching approved proposal."""
        proposal = self._find_approved_proposal(block, source=source, layer=layer)
        if proposal is None:
            return block

        existing_keys = {
            line[2:].split(":", 1)[0].strip().lower()
            for line in block.splitlines()
            if line.startswith("- ") and ":" in line
        }
        additions: list[str] = []
        if "proposal-id" not in existing_keys:
            additions.append(f"- Proposal-ID: {proposal.proposal_id}")
        if "ledger-evidence-ids" not in existing_keys:
            evidence_ids = ", ".join(item.evidence_id for item in proposal.evidence)
            additions.append(f"- Ledger-Evidence-IDs: {evidence_ids}")
        if not additions:
            return block
        return block.rstrip() + "\n" + "\n".join(additions) + "\n"

    def _find_approved_proposal(
        self,
        block: str,
        *,
        source: str,
        layer: MemoryLayer,
    ) -> MemoryProposal | None:
        """Identify the one approved proposal authorized to perform this write."""
        title = extract_title(block)
        if not title:
            return None
        candidates = [
            proposal
            for proposal in self.governance.ledger_for(layer).list_proposals()
            if proposal.status is MemoryProposalStatus.APPROVED
            and proposal.layer == layer
            and proposal.title.casefold() == title.casefold()
            and source == f"governance:{proposal.source}"
            and _normalized_block(proposal.block) == _normalized_block(block)
        ]
        return candidates[0] if len(candidates) == 1 else None


def _normalized_block(block: str) -> str:
    return "\n".join(line.rstrip() for line in block.strip().splitlines())
