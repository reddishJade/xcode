"""Memory 统一检索与兼容性测试。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from xcode.harness.memory import MemoryManager, MemoryRerankPolicy, build_memory_tools


def _block(
    title: str,
    context: str,
    solution: str,
    **metadata: str,
) -> str:
    lines = [
        f"## {title}",
        f"- Context/Query: {context}",
        f"- Solution: {solution}",
        "- Files: src/example.py",
        "- Takeaways: Reuse this verified approach for similar work.",
    ]
    lines.extend(
        f"- {key.replace('_', '-')}: {value}" for key, value in metadata.items()
    )
    return "\n".join(lines) + "\n"


def _manager(
    tmp_path: Path,
    blocks: list[str],
    *,
    embedding_fn: Callable[[str], list[float]] | None = None,
    min_retrieval_score: float = 0.01,
    min_confidence: float = 0.0,
) -> MemoryManager:
    (tmp_path / "MEMORY.md").write_text("\n".join(blocks), encoding="utf-8")
    user_file = tmp_path / "user" / "MEMORY.md"
    user_file.parent.mkdir(exist_ok=True)
    user_file.write_text("", encoding="utf-8")
    return MemoryManager(
        tmp_path,
        user_memory_file=user_file,
        embedding_fn=embedding_fn,
        min_retrieval_score=min_retrieval_score,
        min_confidence=min_confidence,
    )


def test_main_entry_uses_hybrid_score_when_embedding_is_configured(
    tmp_path: Path,
) -> None:
    blocks = [
        _block("Lexical", "alpha beta gamma", "Use lexical handling."),
        _block("Semantic", "alpha beta", "Apply the semantic concept."),
    ]

    def embed(text: str) -> list[float]:
        return (
            [1.0, 0.0]
            if "Semantic" in text or text == "alpha beta gamma"
            else [0.0, 1.0]
        )

    lexical = _manager(tmp_path, blocks).search_memory_records(
        "alpha beta gamma", limit=2, track_usage=False
    )
    hybrid = _manager(tmp_path, blocks, embedding_fn=embed).search_memory_records(
        "alpha beta gamma", limit=2, track_usage=False
    )

    assert {record.title: record.score for record in hybrid} != {
        record.title: record.score for record in lexical
    }
    assert hybrid[0].title == "Semantic"


def test_missing_embedding_uses_bm25_and_compatibility_entry(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        [
            _block("English timeout", "provider timeout retry", "Retry once."),
            _block("Other", "terminal colors", "Use ANSI colors."),
        ],
    )

    main = manager.search_memory_records("provider timeout", track_usage=False)
    compat = manager.hybrid_search_memory_records("provider timeout", track_usage=False)

    assert [record.title for record in main] == [record.title for record in compat]
    assert main[0].title == "English timeout"


def test_search_memory_tool_uses_unified_hybrid_entry(tmp_path: Path) -> None:
    calls = 0

    def embed(text: str) -> list[float]:
        nonlocal calls
        calls += 1
        return [1.0, 0.0] if "Semantic" in text else [0.8, 0.2]

    manager = _manager(
        tmp_path,
        [_block("Semantic tool result", "provider lookup", "Use hybrid retrieval.")],
        embedding_fn=embed,
    )
    tool = build_memory_tools(manager)[0]

    output = tool.handler({"query": "provider lookup"}, None)

    assert calls >= 2
    assert "Semantic tool result" in output


def test_embedding_failure_safely_falls_back_to_bm25(tmp_path: Path) -> None:
    def broken_embedding(_text: str) -> list[float]:
        raise RuntimeError("embedding service unavailable")

    manager = _manager(
        tmp_path,
        [
            _block("中文检索", "修复终端超时重试", "采用指数退避。"),
            _block("无关记录", "颜色主题", "保持高对比度。"),
        ],
        embedding_fn=broken_embedding,
    )

    records = manager.search_memory_records("终端超时", track_usage=False)

    assert records[0].title == "中文检索"


def test_hybrid_results_still_apply_score_and_confidence_gates(tmp_path: Path) -> None:
    def embed(_text: str) -> list[float]:
        return [1.0, 0.0]

    manager = _manager(
        tmp_path,
        [
            _block(
                "Low confidence", "semantic only", "Do not inject.", confidence="0.2"
            ),
            _block("High confidence", "semantic only", "Use this.", confidence="0.9"),
        ],
        embedding_fn=embed,
        min_retrieval_score=0.2,
        min_confidence=0.5,
    )

    records = manager.search_memory_records("different query", track_usage=False)

    assert [record.title for record in records] == ["High confidence"]
    strict = _manager(
        tmp_path,
        [_block("Filtered", "semantic only", "Below score gate.", confidence="0.9")],
        embedding_fn=embed,
        min_retrieval_score=2.0,
    )
    assert strict.search_memory_records("different query", track_usage=False) == []


def test_rerank_penalizes_status_scope_and_historical_failure(tmp_path: Path) -> None:
    policy = MemoryRerankPolicy(freshness_multiplier_min=1.0)
    blocks = [
        _block("Active", "provider timeout", "Use active fix.", scope="cache"),
        _block(
            "Deprecated",
            "provider timeout",
            "Old fix.",
            status="deprecated",
            scope="cache",
        ),
        _block(
            "Review",
            "provider timeout",
            "Review fix.",
            status="needs_review",
            scope="cache",
        ),
        _block("Mismatch", "provider timeout", "Other scope.", scope="database"),
        _block(
            "Failed",
            "provider timeout",
            "Failed before.",
            scope="database",
            failure_count="2",
            success_count="0",
            last_outcome="failure",
        ),
    ]
    manager = _manager(tmp_path, blocks)
    manager.rerank_policy = policy

    records = manager.search_memory_records(
        "provider timeout", scope="cache", limit=10, track_usage=False
    )
    scores = {record.title: record.score for record in records}

    assert records[0].title == "Active"
    assert scores["Deprecated"] < scores["Active"]
    assert scores["Review"] < scores["Active"]
    assert scores["Mismatch"] < scores["Active"]
    assert scores["Failed"] < scores["Active"]


def test_legacy_record_without_new_metadata_still_parses_and_searches(
    tmp_path: Path,
) -> None:
    manager = _manager(
        tmp_path,
        [_block("Legacy parser", "旧格式解析", "继续支持原有字段。")],
    )

    record = manager.search_memory_records("旧格式", track_usage=False)[0]

    assert record.title == "Legacy parser"
    assert record.retrieval_count == 0
    assert record.status == "active"
    assert record.utility == pytest.approx(0.0)
