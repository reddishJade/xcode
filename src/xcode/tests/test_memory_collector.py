"""Tests for typed, scope-separated memory context collection."""

from __future__ import annotations

from pathlib import Path

from xcode.agent.context_assembly import (
    ContextAuthority,
    ContextBlockSource,
    ContextBlockTarget,
    ContextScope,
    ContextTrust,
)
from xcode.agent.context_collector import ContextCollectionInput
from xcode.agent.messages import UserMessage
from xcode.harness.memory import MemoryManager
from xcode.harness.memory.collector import MemoryCollector


def _block(title: str, solution: str, takeaway: str) -> str:
    return (
        f"## {title}\n"
        "- Context/Query: Provider request timeout and retry handling\n"
        f"- Solution: {solution}\n"
        "- Files: src/xcode/provider.py\n"
        f"- Takeaways: {takeaway}\n"
    )


def test_memory_collector_injects_project_memory_as_user_context(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(
        _block(
            "Project retry policy",
            "Retry only transient provider failures with bounded backoff.",
            "Preserve the original failure reason.",
        ),
        layer="project",
    )
    collector = MemoryCollector(manager, project_root=tmp_path)

    blocks = collector.collect(
        ContextCollectionInput(
            project_root=tmp_path,
            messages=[UserMessage(content="How should provider timeout retries work?")],
        )
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block.source is ContextBlockSource.MEMORY
    assert block.target is ContextBlockTarget.USER_CONTEXT
    assert block.authority is ContextAuthority.MEMORY
    assert block.trust is ContextTrust.RUNTIME_INTERNAL
    assert block.scope is ContextScope.REPOSITORY
    assert block.scope_key == str(tmp_path.resolve())
    assert block.provenance.origin == "memory_collector"
    assert block.provenance.locator.startswith("memory:mem_")
    assert "<memory-digest>" in block.content
    assert "Project retry policy" in block.content
    assert "Retry only transient" in block.content

    events = manager.drain_trace_events()
    assert any(event.type == "retrieved" for event in events)
    assert any(event.type == "injected" for event in events)


def test_memory_collector_separates_user_global_memory(tmp_path: Path) -> None:
    manager = MemoryManager(
        tmp_path,
        user_memory_file=tmp_path / "user" / "MEMORY.md",
    )
    assert manager.add_memory_block(
        _block(
            "User retry preference",
            "Prefer concise retry diagnostics across repositories.",
            "Keep retry explanations brief.",
        ),
        layer="user",
    )
    collector = MemoryCollector(manager, project_root=tmp_path)

    blocks = collector.collect(
        ContextCollectionInput(
            project_root=tmp_path,
            messages=[UserMessage(content="Provider timeout retry guidance")],
        )
    )

    assert len(blocks) == 1
    assert blocks[0].scope is ContextScope.USER_GLOBAL
    assert blocks[0].scope_key == "user:global"
    assert "layer=user" in blocks[0].content


def test_memory_collector_skips_empty_or_tiny_user_queries(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    collector = MemoryCollector(manager, project_root=tmp_path)

    assert collector.collect(ContextCollectionInput(project_root=tmp_path)) == []
    assert collector.collect(
        ContextCollectionInput(
            project_root=tmp_path,
            messages=[UserMessage(content="ok")],
        )
    ) == []
