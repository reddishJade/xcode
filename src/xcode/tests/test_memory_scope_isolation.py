"""Cross-repository scope isolation tests for durable memory."""

from __future__ import annotations

from pathlib import Path

from xcode.agent.context_collector import ContextCollectionInput
from xcode.agent.messages import UserMessage
from xcode.harness.memory import MemoryCollector, MemoryManager


def _block(title: str, solution: str) -> str:
    return (
        f"## {title}\n"
        "- Context/Query: Provider timeout retry handling\n"
        f"- Solution: {solution}\n"
        "- Files: src/xcode/provider.py\n"
        "- Takeaways: Preserve the original failure reason.\n"
    )


def test_project_memory_does_not_leak_to_a_different_repository(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    user_memory_file = tmp_path / "user" / "MEMORY.md"
    project_a.mkdir()
    project_b.mkdir()

    manager_a = MemoryManager(project_a, user_memory_file=user_memory_file)
    assert manager_a.add_memory_block(
        _block("Repository A retry convention", "Use the A-only retry strategy."),
        source="repl",
        layer="project",
    )

    manager_b = MemoryManager(project_b, user_memory_file=user_memory_file)
    records = manager_b.search_memory_records("A-only retry strategy", layer="all")

    assert records == []


def test_user_global_memory_crosses_repositories_but_remains_user_context(
    tmp_path: Path,
) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    user_memory_file = tmp_path / "user" / "MEMORY.md"
    project_a.mkdir()
    project_b.mkdir()

    manager_a = MemoryManager(project_a, user_memory_file=user_memory_file)
    assert manager_a.add_memory_block(
        _block("Global retry preference", "Lead with concise retry diagnostics."),
        source="repl",
        layer="user",
    )

    manager_b = MemoryManager(project_b, user_memory_file=user_memory_file)
    blocks = MemoryCollector(manager_b, project_root=project_b).collect(
        ContextCollectionInput(
            project_root=project_b,
            messages=[UserMessage(content="How should timeout retries be explained?")],
        )
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block.scope.value == "user_global"
    assert block.scope_key == "user:global"
    assert "layer=user" in block.content
    assert "Global retry preference" in block.content
    assert "Repository A retry convention" not in block.content
