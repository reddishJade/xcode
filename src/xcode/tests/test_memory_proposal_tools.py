"""Tests for read-only inspection of governed memory proposals."""

from __future__ import annotations

from pathlib import Path

from xcode.harness.memory import MemoryManager, build_memory_tools


def _candidate_block() -> str:
    return (
        "## Compaction proposal\n"
        "- Context/Query: Context assembly boundaries\n"
        "- Solution: Keep workspace instructions outside system authority\n"
        "- Files: src/xcode/agent/context_assembly.py\n"
        "- Takeaways: Review automation-derived memory before promotion\n"
    )


def test_list_memory_proposals_is_read_only_and_shows_evidence(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    manager.consolidate(_candidate_block())
    tools = {tool.name: tool for tool in build_memory_tools(manager)}

    output = tools["list_memory_proposals"].handler({"status": "pending"})

    assert tools["list_memory_proposals"].read_only
    assert "[pending]" in output
    assert "Compaction proposal" in output
    assert "compaction_summary:" in output
    assert not manager.memory_file.exists()


def test_explain_memory_shows_linked_governance_provenance(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_candidate_block(), source="repl", layer="project")
    record = manager.read_memory_records(layer="project")[0]
    tools = {tool.name: tool for tool in build_memory_tools(manager)}

    output = tools["explain_memory"].handler({"memory_id": record.memory_id})

    proposal = manager.governance.ledger.list_proposals()[0]
    assert tools["explain_memory"].read_only
    assert f"proposal={proposal.proposal_id} status=applied" in output
    assert f"ledger_evidence_ids={proposal.evidence[0].evidence_id}" in output
    assert f"evidence[linked] id={proposal.evidence[0].evidence_id}" in output
    assert "trust=trusted_user" in output


def test_explain_memory_marks_legacy_record_without_inventing_provenance(
    tmp_path: Path,
) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_candidate_block(), source="fixture", layer="project")
    record = manager.read_memory_records(layer="project")[0]
    tools = {tool.name: tool for tool in build_memory_tools(manager)}

    output = tools["explain_memory"].handler({"memory_id": record.memory_id})

    assert "governance=legacy_or_untracked" in output
    assert "proposal=" not in output


def test_list_memory_proposals_rejects_unknown_status(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    tool = next(
        item
        for item in build_memory_tools(manager)
        if item.name == "list_memory_proposals"
    )

    output = tool.handler({"status": "unknown"})

    assert output.startswith("status must be one of:")
