"""Read-only memory retrieval and proposal-inspection tools."""

from __future__ import annotations

from typing import cast

from xcode.harness.skills import ToolInput, ToolSpec

from .governance import MemoryLedger, MemoryProposalStatus
from .manager import MemoryLayerFilter, MemoryManager, MemoryRetrievalContext


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

        rendered = [manager.render_search_result(record) for record in records]
        return "\n\n".join(rendered)

    def list_memory_proposals(data: ToolInput) -> str:
        """Render governance proposals without changing approval state."""
        status = str(data.get("status", "pending")).strip().lower() or "pending"
        valid_statuses = {"all", *(item.value for item in MemoryProposalStatus)}
        if status not in valid_statuses:
            allowed = ", ".join(sorted(valid_statuses))
            return f"status must be one of: {allowed}"

        proposals = MemoryLedger(manager.root).list_proposals()
        if status != "all":
            proposals = tuple(
                proposal
                for proposal in proposals
                if proposal.status.value == status
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
                f"{item.kind}:{item.reference} trust={item.trust.value}"
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
