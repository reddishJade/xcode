"""Evidence-backed promotion gate for durable memory writes.

``MemoryManager`` remains the readable project/user MEMORY.md store. This module
adds the separate write-side contract: evidence is recorded first, a proposal is
created, policy decides whether it is rejected, pending approval, or approved,
and only an approved proposal may call ``MemoryManager.add_memory_block``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Literal, Sequence
from uuid import uuid4

from xcode.agent.context_assembly import ContextTrust

from .manager import MemoryLayer, MemoryManager
from .parsing import MemoryEvidence, MemoryType


type MemoryOperation = Literal["add", "update", "retire", "promote_skill"]
type MemoryRequester = Literal["explicit_user", "automation", "consolidation", "agent"]


class MemoryProposalStatus(StrEnum):
    """Lifecycle states for a proposed durable-memory change."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryEvidenceInput:
    """Evidence supplied before a memory proposal can be evaluated."""

    kind: str
    reference: str
    trust: ContextTrust
    scope: str = ""
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class GovernedMemoryEvidence:
    """Evidence after it has been assigned an immutable ledger identity."""

    evidence_id: str
    kind: str
    reference: str
    trust: ContextTrust
    scope: str
    content_hash: str
    created_at: str

    def to_memory_evidence(self) -> MemoryEvidence:
        """Render the compact evidence reference stored in MEMORY.md."""
        return MemoryEvidence(kind=self.kind, reference=self.reference)


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """A pending or resolved write operation against durable memory."""

    proposal_id: str
    operation: MemoryOperation
    title: str
    block: str
    layer: MemoryLayer
    scope: str
    source: str
    requester: MemoryRequester
    memory_type: MemoryType | None
    status: MemoryProposalStatus
    decision_reason: str | None
    created_at: str
    resolved_at: str | None
    applied_memory_id: str | None
    evidence: tuple[GovernedMemoryEvidence, ...]


@dataclass(frozen=True, slots=True)
class MemoryPromotionDecision:
    """Deterministic policy result before any write is applied."""

    status: MemoryProposalStatus
    reason: str

    @property
    def requires_approval(self) -> bool:
        return self.status is MemoryProposalStatus.PENDING


@dataclass(frozen=True, slots=True)
class MemoryProposalResult:
    """Result of proposing, approving, rejecting, or applying a memory change."""

    proposal: MemoryProposal
    decision: MemoryPromotionDecision


class MemoryPromotionPolicy:
    """Conservative first policy for durable-memory promotion.

    Only an explicit user request may auto-approve today. Verified automation is
    retained as a reviewable proposal; this deliberately prevents the old direct
    consolidation path from silently becoming an authority channel before replay
    evaluations exist.
    """

    _REJECTED_TRUST = frozenset(
        {
            ContextTrust.EXTERNAL_UNTRUSTED,
            ContextTrust.WORKSPACE_UNTRUSTED,
        }
    )

    def decide(
        self,
        *,
        operation: MemoryOperation,
        layer: MemoryLayer,
        requester: MemoryRequester,
        evidence: Sequence[GovernedMemoryEvidence],
    ) -> MemoryPromotionDecision:
        if operation != "add":
            return MemoryPromotionDecision(
                MemoryProposalStatus.REJECTED,
                "operation_not_implemented",
            )
        if not evidence:
            return MemoryPromotionDecision(
                MemoryProposalStatus.REJECTED,
                "missing_evidence",
            )
        if any(item.trust in self._REJECTED_TRUST for item in evidence):
            return MemoryPromotionDecision(
                MemoryProposalStatus.REJECTED,
                "untrusted_evidence",
            )
        if requester == "explicit_user":
            return MemoryPromotionDecision(
                MemoryProposalStatus.APPROVED,
                "explicit_user_request",
            )
        if layer == "user":
            return MemoryPromotionDecision(
                MemoryProposalStatus.PENDING,
                "user_layer_requires_explicit_approval",
            )
        return MemoryPromotionDecision(
            MemoryProposalStatus.PENDING,
            "automation_requires_promotion_approval",
        )


class MemoryLedger:
    """Append-only SQLite ledger for memory evidence, proposals, and decisions."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.path = self.project_root / ".local" / "memory" / "governance.sqlite3"

    def record_evidence(
        self,
        items: Sequence[MemoryEvidenceInput],
    ) -> tuple[GovernedMemoryEvidence, ...]:
        now = _utc_now()
        normalized: list[GovernedMemoryEvidence] = []
        with self._connect() as connection:
            for item in items:
                evidence = GovernedMemoryEvidence(
                    evidence_id=f"ev_{uuid4().hex}",
                    kind=item.kind.strip() or "unknown",
                    reference=item.reference.strip(),
                    trust=item.trust,
                    scope=item.scope.strip(),
                    content_hash=item.content_hash.strip(),
                    created_at=now,
                )
                if not evidence.reference:
                    raise ValueError("memory evidence reference is required")
                connection.execute(
                    """
                    INSERT INTO memory_evidence (
                        evidence_id, kind, reference, trust, scope, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence.evidence_id,
                        evidence.kind,
                        evidence.reference,
                        evidence.trust.value,
                        evidence.scope,
                        evidence.content_hash,
                        evidence.created_at,
                    ),
                )
                normalized.append(evidence)
        return tuple(normalized)

    def create_proposal(
        self,
        *,
        operation: MemoryOperation,
        title: str,
        block: str,
        layer: MemoryLayer,
        scope: str,
        source: str,
        requester: MemoryRequester,
        memory_type: MemoryType | None,
        evidence: Sequence[GovernedMemoryEvidence],
    ) -> MemoryProposal:
        proposal_id = f"mp_{uuid4().hex}"
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_proposals (
                    proposal_id, operation, title, block, layer, scope, source, requester,
                    memory_type, status, decision_reason, created_at, resolved_at,
                    applied_memory_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
                """,
                (
                    proposal_id,
                    operation,
                    title,
                    block,
                    layer,
                    scope,
                    source,
                    requester,
                    memory_type,
                    MemoryProposalStatus.PENDING.value,
                    created_at,
                ),
            )
            for item in evidence:
                connection.execute(
                    """
                    INSERT INTO memory_proposal_evidence (proposal_id, evidence_id)
                    VALUES (?, ?)
                    """,
                    (proposal_id, item.evidence_id),
                )
            self._append_event(
                connection,
                proposal_id,
                "created",
                {"requester": requester, "source": source},
            )
        return self.get_proposal(proposal_id)

    def get_proposal(self, proposal_id: str) -> MemoryProposal:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown memory proposal: {proposal_id}")
            evidence_rows = connection.execute(
                """
                SELECT e.*
                FROM memory_evidence AS e
                JOIN memory_proposal_evidence AS pe ON pe.evidence_id = e.evidence_id
                WHERE pe.proposal_id = ?
                ORDER BY e.created_at, e.evidence_id
                """,
                (proposal_id,),
            ).fetchall()
        return _proposal_from_row(row, evidence_rows)

    def list_proposals(self) -> tuple[MemoryProposal, ...]:
        with self._connect() as connection:
            proposal_rows = connection.execute(
                "SELECT proposal_id FROM memory_proposals ORDER BY created_at, proposal_id"
            ).fetchall()
        return tuple(self.get_proposal(str(row["proposal_id"])) for row in proposal_rows)

    def resolve(
        self,
        proposal_id: str,
        *,
        status: MemoryProposalStatus,
        reason: str,
        applied_memory_id: str | None = None,
    ) -> MemoryProposal:
        if status not in {
            MemoryProposalStatus.APPROVED,
            MemoryProposalStatus.REJECTED,
            MemoryProposalStatus.APPLIED,
            MemoryProposalStatus.FAILED,
        }:
            raise ValueError(f"unsupported proposal resolution status: {status}")
        with self._connect() as connection:
            current = connection.execute(
                "SELECT status FROM memory_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown memory proposal: {proposal_id}")
            _validate_transition(
                MemoryProposalStatus(str(current["status"])),
                status,
            )
            connection.execute(
                """
                UPDATE memory_proposals
                SET status = ?, decision_reason = ?, resolved_at = ?, applied_memory_id = ?
                WHERE proposal_id = ?
                """,
                (status.value, reason, _utc_now(), applied_memory_id, proposal_id),
            )
            self._append_event(
                connection,
                proposal_id,
                status.value,
                {"reason": reason, "applied_memory_id": applied_memory_id},
            )
        return self.get_proposal(proposal_id)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_evidence (
                evidence_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                reference TEXT NOT NULL,
                trust TEXT NOT NULL,
                scope TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_proposals (
                proposal_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                title TEXT NOT NULL,
                block TEXT NOT NULL,
                layer TEXT NOT NULL,
                scope TEXT NOT NULL,
                source TEXT NOT NULL,
                requester TEXT NOT NULL,
                memory_type TEXT,
                status TEXT NOT NULL,
                decision_reason TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                applied_memory_id TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_proposal_evidence (
                proposal_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                PRIMARY KEY (proposal_id, evidence_id),
                FOREIGN KEY (proposal_id) REFERENCES memory_proposals(proposal_id),
                FOREIGN KEY (evidence_id) REFERENCES memory_evidence(evidence_id)
            );

            CREATE TABLE IF NOT EXISTS memory_governance_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (proposal_id) REFERENCES memory_proposals(proposal_id)
            );
            """
        )

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        proposal_id: str,
        event_type: str,
        details: dict[str, object],
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_governance_events (
                proposal_id, event_type, details_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                proposal_id,
                event_type,
                json.dumps(details, sort_keys=True, ensure_ascii=False),
                _utc_now(),
            ),
        )


class MemoryGovernance:
    """Coordinates ledger recording, policy evaluation, and durable writes."""

    def __init__(
        self,
        project_root: Path,
        manager: MemoryManager | None = None,
        policy: MemoryPromotionPolicy | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manager = manager or MemoryManager(self.project_root)
        self.ledger = MemoryLedger(self.project_root)
        self.policy = policy or MemoryPromotionPolicy()

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
        recorded_evidence = self.ledger.record_evidence(evidence)
        proposal = self.ledger.create_proposal(
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
            proposal = self.ledger.resolve(
                proposal.proposal_id,
                status=MemoryProposalStatus.REJECTED,
                reason=decision.reason,
            )
            return MemoryProposalResult(proposal=proposal, decision=decision)
        if decision.status is MemoryProposalStatus.APPROVED:
            proposal = self.ledger.resolve(
                proposal.proposal_id,
                status=MemoryProposalStatus.APPROVED,
                reason=decision.reason,
            )
            proposal = self.apply(proposal.proposal_id)
            return MemoryProposalResult(proposal=proposal, decision=decision)
        return MemoryProposalResult(proposal=proposal, decision=decision)

    def approve(self, proposal_id: str, *, approver: str = "user") -> MemoryProposal:
        proposal = self.ledger.get_proposal(proposal_id)
        if proposal.status is not MemoryProposalStatus.PENDING:
            raise ValueError(f"proposal is not pending: {proposal.status}")
        approved = self.ledger.resolve(
            proposal_id,
            status=MemoryProposalStatus.APPROVED,
            reason=f"approved_by:{approver}",
        )
        return self.apply(approved.proposal_id)

    def reject(self, proposal_id: str, *, reason: str = "rejected_by_user") -> MemoryProposal:
        proposal = self.ledger.get_proposal(proposal_id)
        if proposal.status is not MemoryProposalStatus.PENDING:
            raise ValueError(f"proposal is not pending: {proposal.status}")
        return self.ledger.resolve(
            proposal_id,
            status=MemoryProposalStatus.REJECTED,
            reason=reason,
        )

    def apply(self, proposal_id: str) -> MemoryProposal:
        proposal = self.ledger.get_proposal(proposal_id)
        if proposal.status is not MemoryProposalStatus.APPROVED:
            raise ValueError(f"proposal is not approved: {proposal.status}")
        if proposal.operation != "add":
            return self.ledger.resolve(
                proposal_id,
                status=MemoryProposalStatus.FAILED,
                reason="operation_not_implemented",
            )

        validity = _validity_for_evidence(proposal.evidence)
        persisted = self.manager.add_memory_block(
            proposal.block,
            source=f"governance:{proposal.source}",
            scope=proposal.scope,
            memory_type=proposal.memory_type,
            status="active",
            validity=validity,
            evidence=tuple(item.to_memory_evidence() for item in proposal.evidence),
            layer=proposal.layer,
        )
        if not persisted:
            return self.ledger.resolve(
                proposal_id,
                status=MemoryProposalStatus.FAILED,
                reason="memory_manager_rejected_candidate",
            )

        memory_id = _memory_id_for_title(self.manager, proposal.layer, proposal.title)
        return self.ledger.resolve(
            proposal_id,
            status=MemoryProposalStatus.APPLIED,
            reason="applied_to_memory_manager",
            applied_memory_id=memory_id,
        )

    def add_explicit_user_memory(
        self,
        *,
        block: str,
        title: str,
        layer: MemoryLayer,
        scope: str | None = None,
        source: str = "repl",
        memory_type: MemoryType | None = None,
    ) -> MemoryProposalResult:
        """Record a user's explicit `/memory add` as evidence-backed promotion."""
        content_hash = sha256(block.encode("utf-8")).hexdigest()
        project_scope = scope or str(self.project_root)
        return self.propose(
            block=block,
            title=title,
            layer=layer,
            scope=project_scope,
            source=source,
            requester="explicit_user",
            memory_type=memory_type,
            evidence=(
                MemoryEvidenceInput(
                    kind="user_request",
                    reference=f"{source}:memory-add:{content_hash[:16]}",
                    trust=ContextTrust.TRUSTED_USER,
                    scope=project_scope,
                    content_hash=content_hash,
                ),
            ),
        )


def _proposal_from_row(
    row: sqlite3.Row,
    evidence_rows: Sequence[sqlite3.Row],
) -> MemoryProposal:
    evidence = tuple(
        GovernedMemoryEvidence(
            evidence_id=str(item["evidence_id"]),
            kind=str(item["kind"]),
            reference=str(item["reference"]),
            trust=ContextTrust(str(item["trust"])),
            scope=str(item["scope"]),
            content_hash=str(item["content_hash"]),
            created_at=str(item["created_at"]),
        )
        for item in evidence_rows
    )
    raw_memory_type = row["memory_type"]
    return MemoryProposal(
        proposal_id=str(row["proposal_id"]),
        operation=str(row["operation"]),  # type: ignore[arg-type]
        title=str(row["title"]),
        block=str(row["block"]),
        layer=str(row["layer"]),  # type: ignore[arg-type]
        scope=str(row["scope"]),
        source=str(row["source"]),
        requester=str(row["requester"]),  # type: ignore[arg-type]
        memory_type=str(raw_memory_type) if raw_memory_type else None,  # type: ignore[arg-type]
        status=MemoryProposalStatus(str(row["status"])),
        decision_reason=str(row["decision_reason"]) if row["decision_reason"] else None,
        created_at=str(row["created_at"]),
        resolved_at=str(row["resolved_at"]) if row["resolved_at"] else None,
        applied_memory_id=str(row["applied_memory_id"])
        if row["applied_memory_id"]
        else None,
        evidence=evidence,
    )


def _validate_transition(
    current: MemoryProposalStatus,
    next_status: MemoryProposalStatus,
) -> None:
    allowed: dict[MemoryProposalStatus, frozenset[MemoryProposalStatus]] = {
        MemoryProposalStatus.PENDING: frozenset(
            {MemoryProposalStatus.APPROVED, MemoryProposalStatus.REJECTED}
        ),
        MemoryProposalStatus.APPROVED: frozenset(
            {MemoryProposalStatus.APPLIED, MemoryProposalStatus.FAILED}
        ),
        MemoryProposalStatus.REJECTED: frozenset(),
        MemoryProposalStatus.APPLIED: frozenset(),
        MemoryProposalStatus.FAILED: frozenset(),
    }
    if next_status not in allowed[current]:
        raise ValueError(f"invalid memory proposal transition: {current} -> {next_status}")


def _validity_for_evidence(evidence: Sequence[GovernedMemoryEvidence]) -> str:
    trusts = {item.trust for item in evidence}
    if trusts == {ContextTrust.VERIFIED_TOOL}:
        return "verified"
    if ContextTrust.TRUSTED_USER in trusts:
        return "user_asserted"
    return "needs_review"


def _memory_id_for_title(manager: MemoryManager, layer: MemoryLayer, title: str) -> str | None:
    for record in manager.read_memory_records(layer=layer):
        if record.title.casefold() == title.casefold():
            return record.memory_id
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
