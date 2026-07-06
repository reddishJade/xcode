"""Regression tests for layer-aware durable-memory governance."""

from __future__ import annotations

from pathlib import Path

from xcode.harness.memory import MemoryManager, build_memory_tools


def _block() -> str:
    return (
        "## Global response preference\n"
        "- Context/Query: Explain technical trade-offs\n"
        "- Solution: Lead with the recommendation and keep the rationale concise.\n"
        "- Files: (user preference)\n"
        "- Takeaways: Prefer direct answers across repositories.\n"
    )


def test_user_memory_ledger_survives_repository_change(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    user_memory_file = tmp_path / "user" / "MEMORY.md"
    project_a.mkdir()
    project_b.mkdir()

    manager_a = MemoryManager(project_a, user_memory_file=user_memory_file)
    assert manager_a.add_memory_block(_block(), source="repl", layer="user")
    proposal = manager_a.governance.ledger_for("user").list_proposals()[0]
    assert proposal.scope == "user_global"
    assert manager_a.governance.ledger.path != manager_a.governance.ledger_for("user").path

    manager_b = MemoryManager(project_b, user_memory_file=user_memory_file)
    record = manager_b.read_memory_records(layer="user")[0]
    tools = {tool.name: tool for tool in build_memory_tools(manager_b)}

    output = tools["explain_memory"].handler({"memory_id": record.memory_id})
    listed = tools["list_memory_proposals"].handler({"status": "applied"})

    assert f"proposal={proposal.proposal_id} status=applied" in output
    assert f"evidence[linked] id={proposal.evidence[0].evidence_id}" in output
    assert proposal.proposal_id in listed
    assert "Global response preference" in listed
