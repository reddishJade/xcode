"""Memory prompt 注入预算测试。"""

from __future__ import annotations

from pathlib import Path

from xcode.agent._compaction import estimate_tokens
from xcode.harness.agent_runtime.prompting.builder import _render_memory_context
from xcode.harness.memory import MemoryManager


def _block(title: str, context: str, solution: str, memory_type: str) -> str:
    return (
        f"## {title}\n"
        f"- Context/Query: {context}\n"
        f"- Solution: {solution}\n"
        "- Files: src/example.py\n"
        "- Takeaways: Keep the query-specific result first.\n"
        f"- Memory-Type: {memory_type}\n"
    )


def test_prompt_budget_uses_query_order_and_keeps_relevant_episodic(
    tmp_path: Path,
) -> None:
    relevant = _block(
        "Relevant episodic",
        "rare widget timeout exact phrase",
        "EPISODIC-SOLUTION",
        "episodic",
    )
    global_important = _block(
        "Generic semantic",
        "widget background",
        "SEMANTIC-SOLUTION",
        "semantic",
    )
    (tmp_path / "MEMORY.md").write_text(
        relevant + "\n" + global_important, encoding="utf-8"
    )
    user_file = tmp_path / "user-memory.md"
    user_file.write_text("", encoding="utf-8")
    manager = MemoryManager(
        tmp_path,
        user_memory_file=user_file,
        min_retrieval_score=0.0,
    )
    ranked = manager.search_memory_records(
        "rare widget timeout exact phrase", limit=10, track_usage=False
    )
    budget = estimate_tokens(manager.render_prompt_packet(ranked[0]))

    rendered = _render_memory_context(
        manager,
        "rare widget timeout exact phrase",
        max_tokens=budget,
    )

    assert "EPISODIC-SOLUTION" in rendered
    assert "SEMANTIC-SOLUTION" not in rendered


def test_oversized_top_record_is_not_replaced_by_lower_ranked_record(
    tmp_path: Path,
) -> None:
    (tmp_path / "MEMORY.md").write_text("", encoding="utf-8")
    manager = MemoryManager(tmp_path, user_memory_file=tmp_path / "user-memory.md")
    first = _block("First", "top match", "X" * 500, "episodic")
    second = _block("Second", "weak match", "short", "semantic")
    (tmp_path / "MEMORY.md").write_text(first + "\n" + second, encoding="utf-8")
    records = manager.read_memory_records(layer="project")

    selected = manager.select_budgeted_records(records, max_tokens=20)

    assert selected == []
