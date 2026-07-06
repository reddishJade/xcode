"""Read-only integrity checks for governed durable memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .governance import MemoryLedger, MemoryProposalStatus
from .governed_manager import GovernedMemoryManager
from .manager import MemoryLayer, MemoryManager
from .parsing import MemoryRecord


type MemoryIntegrityCode = Literal[
    "missing_proposal",
    "proposal_not_applied",
    "layer_mismatch",
    "scope_mismatch",
    "user_scope_violation",
    "memory_id_mismatch",
    "evidence_mismatch",
]


@dataclass(frozen=True, slots=True)
class MemoryIntegrityIssue:
    """One non-destructive mismatch between durable memory and its ledger."""

    code: MemoryIntegrityCode
    memory_id: str
    layer: str
    detail: str


@dataclass(frozen=True, slots=True)
class MemoryIntegrityReport:
    """Summary of governed-memory provenance checks."""

    checked_records: int
    governed_records: int
    legacy_records: int
    issues: tuple[MemoryIntegrityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def audit_memory_integrity(manager: MemoryManager) -> MemoryIntegrityReport:
    """Verify that governed records have an applied, same-layer evidence trail.

    Records without a proposal id are intentionally reported as legacy rather than
    failures: pre-governance memory remains readable but is not silently promoted
    to evidence-backed status.
    """
    issues: list[MemoryIntegrityIssue] = []
    records = manager.read_memory_records(layer="all")
    governed_records = 0
    legacy_records = 0

    for record in records:
        proposal_id = record.fields.get("proposal-id", "").strip()
        if not proposal_id:
            legacy_records += 1
            continue
        governed_records += 1
        ledger = _ledger_for(manager, record.layer)
        try:
            proposal = ledger.get_proposal(proposal_id)
        except KeyError:
            issues.append(
                _issue(
                    "missing_proposal",
                    record,
                    f"proposal_id={proposal_id} is absent from the {record.layer} ledger",
                )
            )
            continue

        if proposal.status is not MemoryProposalStatus.APPLIED:
            issues.append(
                _issue(
                    "proposal_not_applied",
                    record,
                    f"proposal_id={proposal_id} status={proposal.status.value}",
                )
            )
        if proposal.layer != record.layer:
            issues.append(
                _issue(
                    "layer_mismatch",
                    record,
                    f"proposal_layer={proposal.layer} record_layer={record.layer}",
                )
            )
        if proposal.scope != (record.scope or ""):
            issues.append(
                _issue(
                    "scope_mismatch",
                    record,
                    f"proposal_scope={proposal.scope!r} record_scope={record.scope!r}",
                )
            )
        if record.layer == "user" and record.scope != "user_global":
            issues.append(
                _issue(
                    "user_scope_violation",
                    record,
                    f"user record scope must be user_global, got {record.scope!r}",
                )
            )
        if proposal.applied_memory_id and proposal.applied_memory_id != record.memory_id:
            issues.append(
                _issue(
                    "memory_id_mismatch",
                    record,
                    f"proposal_memory_id={proposal.applied_memory_id} record_memory_id={record.memory_id}",
                )
            )

        expected_evidence = tuple(item.evidence_id for item in proposal.evidence)
        actual_evidence = _csv_values(record.fields.get("ledger-evidence-ids", ""))
        if actual_evidence != expected_evidence:
            issues.append(
                _issue(
                    "evidence_mismatch",
                    record,
                    f"proposal_evidence={expected_evidence} record_evidence={actual_evidence}",
                )
            )

    return MemoryIntegrityReport(
        checked_records=len(records),
        governed_records=governed_records,
        legacy_records=legacy_records,
        issues=tuple(issues),
    )


def _ledger_for(manager: MemoryManager, layer: str) -> MemoryLedger:
    if isinstance(manager, GovernedMemoryManager) and layer in {"project", "user"}:
        return manager.governance.ledger_for(cast(MemoryLayer, layer))
    return MemoryLedger(manager.root)


def _issue(
    code: MemoryIntegrityCode,
    record: MemoryRecord,
    detail: str,
) -> MemoryIntegrityIssue:
    return MemoryIntegrityIssue(
        code=code,
        memory_id=record.memory_id,
        layer=record.layer,
        detail=detail,
    )


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())
