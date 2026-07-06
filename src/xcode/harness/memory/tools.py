"""Read-only memory retrieval, provenance, and proposal-inspection tools."""

from __future__ import annotations

from typing import cast

from xcode.harness.skills import ToolInput, ToolSpec

from .governance import MemoryLedger, MemoryProposalStatus
from .manager import MemoryLayerFilter, MemoryManager, MemoryRetrievalContext
from .parsing import MemoryRecord


def build_memory_tools(manager: MemoryManager) -> tuple[ToolSpec, ...]:
    """Build opt-in memory tools without exposing proposal approval to the agent."""

    def search_memory(data: ToolInput) -> str:
        """Search project and user memory and render source metadata."""
        query = str(data.get("query", "")).strip()
        if not query:
            return "query is required"
        limit = _parse_limit(data.get("limit", 3))
        scope = _optional_text(data.get("scope"))
        current_file = _optional_text(data.get("current_file"))
        task_phase = _optional_text(data.get("task_phase"))
        layer = str(data.get("layer", "all"))
        if layer not in {"all", "project", "user"}:
            return "layer must be one of: all, project, user"
        symbols = _optional_list(data.get("symbols"))
        error_messages = _optional_list(data.get("error_messages"))
        modules = _optional_list(data.get("modules"))
        recent_files = _optional_list(data.get("recent_files"))

        records = manager.search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=cast(MemoryLayerFilter, layer),
            source="tool",
            retrieval_context=MemoryRetrievalContext(
                query=query,
                scope=scope,
                current_file=current_file,
                symbols=symbols,
                error_messages=error_messages,
                task_phase=task_phase,
                modules=modules,
                recent_files=recent_files,
            ),
        )
        if not records:
            return f"No memory matching {query!r}."
        return "\n\n".join(manager.render_search_result(record) for record in records)

    def explain_memory(data: ToolInput) -> str:
        """Explain one durable memory record and its governed provenance."""
        memory_id = str(data.get("memory_id", "")).strip()
        if not memory_id:
            return "memory_id is required"
        record = _find_record(manager, memory_id)
        if record is None:
            return f"Unknown memory: {memory_id}"

        lines = [
            f"Memory: {record.memory_id}",
            f"title={record.title}",
            f"layer={record.layer} scope={record.scope or '(unscoped)'}",
            f"status={record.status} validity={record.validity}",
        ]
        proposal_id = record.fields.get("proposal-id", "").strip()
        evidence_ids = _csv_values(record.fields.get("ledger-evidence-ids", ""))
        if not proposal_id:
            lines.append("governance=legacy_or_untracked")
            return "\n".join(lines)

        proposal = _find_proposal(MemoryLedger(manager.root), proposal_id)
        if proposal is None:
            lines.append(f"proposal={proposal_id} status=missing_from_ledger")
            return "\n".join(lines)

        lines.append(
            f"proposal={proposal.proposal_id} status={proposal.status.value} "
            f"source={proposal.source} requester={proposal.requester}"
        )
        lines.append(f"proposal_scope={proposal.scope}")
        if proposal.decision_reason:
            lines.append(f"decision={proposal.decision_reason}")
        if evidence_ids:
            lines.append("ledger_evidence_ids=" + ", ".join(evidence_ids))
        for item in proposal.evidence:
            marker = "linked" if item.evidence_id in evidence_ids else "unlinked"
            lines.append(
                f"evidence[{marker}] id={item.evidence_id} kind={item.kind} "
                f"trust={item.trust.value} reference={item.reference}"
            )
        return "\n".join(lines)

    def list_memory_proposals(data: ToolInput) -> str:
        """Render governance proposals without changing approval state."""
        status = str(data.get("status", "pending")).strip().lower() or "pending"
        valid_statuses = {"all", *(item.value for item in MemoryProposalStatus)}
        if status not in valid_statuses:
            return "status must be one of: " + ", ".join(sorted(valid_statuses))

        proposals = MemoryLedger(manager.root).list_proposals()
        if status != "all":
            proposals = tuple(
                proposal for proposal in proposals if proposal.status.value == status
            )
        if not proposals:
            label = "matching" if status == "all" else status
            return f"No {label} memory proposals."

        lines = [f"Memory proposals ({len(proposals)}):"]
        for proposal in proposals:
            lines.append(
                f"- [{proposal.status.value}] {proposal.proposal_id}: {proposal.title}"
            )
            lines.append(
                f"  operation={proposal.operation} layer={proposal.layer} "
                f"source={proposal.source} requester={proposal.requester}"
            )
            if proposal.scope:
                lines.append(f"  scope={proposal.scope}")
            if proposal.decision_reason:
                lines.append(f"  decision={proposal.decision_reason}")
            evidence = "; ".join(
                f"{item.evidence_id} {item.kind}:{item.reference} trust={item.trust.value}"
                for item in proposal.evidence
            )
            lines.append(f"  evidence={evidence or '(none)'}")
        return "\n".join(lines)

    return (
        ToolSpec(
            name="search_memory",
            description=(
                "Search project and user memory for prior solutions, constraints, "
                "files, and takeaways relevant to the current task."
            ),
            input_hint=(
                'JSON: {"query": "provider timeout", "limit": 3, '
                '"scope": "providers", "current_file": "src/provider.py", '
                '"symbols": ["ProviderClient"], "error_messages": ["connection timeout"], '
                '"task_phase": "debug", "modules": ["providers"], '
                '"recent_files": ["src/provider.py"], "layer": "all"}'
            ),
            handler=search_memory,
            schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language memory search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional scope used to rerank matching records.",
                    },
                    "current_file": {
                        "type": "string",
                        "description": "Current file relevant to the task.",
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant code symbols for retrieval and reranking.",
                    },
                    "error_messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant error messages or failure signatures.",
                    },
                    "task_phase": {
                        "type": "string",
                        "description": "Current task phase, such as debug, implement, or verify.",
                    },
                    "modules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant project modules or subsystems.",
                    },
                    "recent_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recent files already made relevant by the task.",
                    },
                    "layer": {
                        "type": "string",
                        "enum": ["all", "project", "user"],
                        "default": "all",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            read_only=True,
            group="memory",
            prompt_snippet=(
                "Search opt-in project and user memory when prior decisions or "
                "solutions may affect the task."
            ),
        ),
        ToolSpec(
            name="explain_memory",
            description=(
                "Explain one memory record's scope, governance proposal, and "
                "evidence provenance. This tool is read-only."
            ),
            input_hint='JSON: {"memory_id": "mem_abcd1234"}',
            handler=explain_memory,
            schema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The mem_* identifier returned by a memory digest or search.",
                    }
                },
                "required": ["memory_id"],
                "additionalProperties": False,
            },
            read_only=True,
            group="memory",
            prompt_snippet=(
                "Use explain_memory when a retrieved memory could materially affect "
                "the task and its provenance or scope needs verification."
            ),
        ),
        ToolSpec(
            name="list_memory_proposals",
            description=(
                "List evidence-backed memory proposals and their approval state. "
                "This tool is read-only and cannot approve or apply a proposal."
            ),
            input_hint='JSON: {"status": "pending"}',
            handler=list_memory_proposals,
            schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "all",
                            "pending",
                            "approved",
                            "rejected",
                            "applied",
                            "failed",
                        ],
                        "default": "pending",
                    }
                },
                "additionalProperties": False,
            },
            read_only=True,
            group="memory",
            prompt_snippet=(
                "Inspect pending memory proposals when a prior learning may need "
                "human review. Do not treat a pending proposal as durable memory."
            ),
        ),
    )


def _find_record(manager: MemoryManager, memory_id: str) -> MemoryRecord | None:
    for record in manager.read_memory_records(layer="all"):
        if record.memory_id == memory_id:
            return record
    return None


def _find_proposal(ledger: MemoryLedger, proposal_id: str) -> object | None:
    try:
        return ledger.get_proposal(proposal_id)
    except KeyError:
        return None


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_limit(value: object) -> int:
    """Constrain user input to a safe result range."""
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return 3
    else:
        return 3
    return min(max(parsed, 1), 10)


def _optional_text(value: object) -> str | None:
    """Normalize optional input to non-empty text."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_list(value: object) -> tuple[str, ...]:
    """Normalize optional input to a tuple of text values."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value]
        return tuple(item for item in normalized if item)
    return ()
