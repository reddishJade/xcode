from __future__ import annotations

import tempfile
from pathlib import Path

from xcode.harness.memory import (
    MemoryJudgeResult,
    MemoryManager,
)
import pytest


class TestMemoryConsolidationHook:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manager = MemoryManager(self.root)

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_consolidate_extracts_decision_from_structured_summary(self) -> None:
        """验证 consolidate 从 Key Decisions 节提取决策并写入记忆。"""
        summary = (
            "[Compressed]\n"
            "## Goal\n"
            "Build a reliable memory system\n"
            "## Key Decisions\n"
            "- Use BM25 for lexical search: balances speed and relevance\n"
            "## Next Steps\n"
            "Add semantic embedding\n"
        )
        self.manager.consolidate(summary)

        assert self.manager.memory_file.exists()
        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        assert "Decision: Use BM25 for lexical search" in memory_text
        trace_events = self.manager.drain_trace_events()
        assert [event.type for event in trace_events] == [
            "candidate_created",
            "accepted",
        ]

    def test_consolidate_rejects_ephemeral_session_only_decision(self) -> None:
        """验证 consolidate 中带有临场标记的决策被 _has_reusable_scope 拒绝。"""
        summary = (
            "[Compressed]\n"
            "## Key Decisions\n"
            "- Use temp file for this session only: scratch data\n"
        )
        self.manager.consolidate(summary)

        assert not self.manager.memory_file.exists()
        archive_files = list(self.manager.archive_dir.glob("*.md"))
        assert len(archive_files) == 1
        trace_events = self.manager.drain_trace_events()
        assert [event.type for event in trace_events] == ["rejected"]
        assert trace_events[0].rejection_reason == "scope_gate_failed"

    def test_llm_judge_rejects_low_value_decision(self) -> None:
        """验证 judge_fn 可以拒绝低价值决策。"""

        def reject_judge(decision_text: str) -> MemoryJudgeResult:
            return MemoryJudgeResult(
                is_worth_remembering=False,
                confidence=0.1,
                reasoning="test rejection",
            )

        self.manager.set_judge_fn(reject_judge)
        decision = "Obvious fix: add type annotation"
        block = self.manager._decision_to_memory_block(decision)
        assert block is None
        trace_events = self.manager.drain_trace_events()
        assert any(e.rejection_reason == "llm_judge_rejected" for e in trace_events)

    def test_llm_judge_accepts_valuable_decision(self) -> None:
        """验证 judge_fn 可以接受高价值决策并生成结构化内容。"""

        def accept_judge(decision_text: str) -> MemoryJudgeResult:
            return MemoryJudgeResult(
                is_worth_remembering=True,
                confidence=0.9,
                scope="provider",
                related_files=("src/provider.py",),
                suggested_title="Decision: Use streaming protocol",
                suggested_context="Need async streaming for provider API",
                suggested_solution="Use StreamProvider protocol with async generators",
                suggested_takeaways="Streaming improves latency by 40%",
                reasoning="Architecture decision with lasting impact",
            )

        self.manager.set_judge_fn(accept_judge)
        decision = "Use streaming protocol for provider API"
        block = self.manager._decision_to_memory_block(decision)
        assert block is not None
        assert "Decision: Use streaming protocol" in block
        assert "- Scope: provider" in block
        assert "src/provider.py" in block
        assert "- Confidence: 0.90" in block
        trace_events = self.manager.drain_trace_events()
        assert not any(e.rejection_reason == "llm_judge_rejected" for e in trace_events)

    def test_template_fallback_when_no_judge_fn(self) -> None:
        """验证未配置 judge_fn 时回退到纯模板路径。"""
        self.manager.set_judge_fn(None)
        decision = "Use Redis for caching: improves response time by 50%"
        block = self.manager._decision_to_memory_block(decision)
        assert block is not None
        assert "Decision: Use Redis for caching" in block
        assert "improves response time" in block
        assert "(see project)" in block

    def test_consolidate_with_llm_judge_filters_decisions(self) -> None:
        """验证 consolidate 使用 judge_fn 过滤并结构化决策。"""

        def picky_judge(decision_text: str) -> MemoryJudgeResult:
            # 只接受含有 "architecture" 的决策
            if "architecture" in decision_text.lower():
                return MemoryJudgeResult(
                    is_worth_remembering=True,
                    confidence=0.85,
                    suggested_title="Decision: " + decision_text[:40],
                    suggested_context=decision_text,
                    suggested_solution=decision_text,
                    suggested_takeaways=decision_text,
                )
            return MemoryJudgeResult(
                is_worth_remembering=False,
                confidence=0.2,
                reasoning="not architecture-relevant",
            )

        self.manager.set_judge_fn(picky_judge)
        summary = (
            "[Compressed]\n"
            "## Key Decisions\n"
            "- Use Redis for caching: improves performance\n"
            "- Adopt layered architecture: separates concerns\n"
            "- Pin pytest version to 8.0\n"
        )
        self.manager.consolidate(summary)

        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        # 只有 architecture 决策被接受
        assert "Adopt layered architecture" in memory_text
        assert "Use Redis for caching" not in memory_text
        assert "Pin pytest version" not in memory_text
        trace_events = self.manager.drain_trace_events()
        accepted = [e for e in trace_events if e.type == "accepted"]
        rejected = [e for e in trace_events if e.type == "rejected"]
        assert len(accepted) == 1
        assert len(rejected) == 2

    def test_hybrid_search_falls_back_when_no_embedding(self) -> None:
        """验证未配置 embedding_fn 时 hybrid search 降级为 BM25。"""
        self.manager.set_judge_fn(None)

        # 写入一条记忆
        block = (
            "## Test hybrid\n"
            "- Context/Query: hybrid search test\n"
            "- Solution: works with BM25 only\n"
            "- Files: test.py\n"
            "- Takeaways: verify fallback\n"
        )
        self.manager.add_memory_block(block)

        results = self.manager.hybrid_search_memory_records(
            "hybrid search test", limit=3
        )
        assert len(results) == 1
        assert results[0].title == "Test hybrid"

    def test_hybrid_search_with_embedding_rerank(self) -> None:
        """验证配置 embedding_fn 时 hybrid search 使用语义重排。"""

        def dummy_embed(text: str) -> list[float]:
            # 简单模拟：关键词匹配式 embedding
            words = text.lower().split()
            vec = [0.0] * 10
            for i, w in enumerate(words):
                vec[i % 10] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            return [x / norm for x in vec] if norm > 0 else vec

        self.manager.set_embedding_fn(dummy_embed)

        block = (
            "## Test embedding\n"
            "- Context/Query: embedding search test\n"
            "- Solution: works with dummy embedding\n"
            "- Files: test.py\n"
            "- Takeaways: verify embedding path\n"
        )
        self.manager.add_memory_block(block)

        results = self.manager.hybrid_search_memory_records(
            "embedding search test", limit=3
        )
        assert len(results) >= 1
        assert results[0].title == "Test embedding"

    def test_set_judge_fn_and_embedding_fn_late_binding(self) -> None:
        """验证 set_judge_fn / set_embedding_fn 运行时绑定。"""
        assert self.manager.judge_fn is None  # 默认无
        assert self.manager.embedding_fn is None

        def mock_judge(text: str) -> MemoryJudgeResult:
            return MemoryJudgeResult(is_worth_remembering=True, confidence=0.8)

        self.manager.set_judge_fn(mock_judge)
        assert self.manager.judge_fn is not None

        self.manager.set_embedding_fn(lambda t: [0.5, 0.5])
        assert self.manager.embedding_fn is not None

        self.manager.set_judge_fn(None)
        self.manager.set_embedding_fn(None)
        assert self.manager.judge_fn is None
        assert self.manager.embedding_fn is None

    def test_llm_judge_short_text_rejected(self) -> None:
        """验证过短的决策文本被  judge_fn 拒绝（不调用 LLM）。"""
        call_count = 0

        def judge(decision_text: str) -> MemoryJudgeResult:
            nonlocal call_count
            call_count += 1
            return MemoryJudgeResult(is_worth_remembering=True, confidence=1.0)

        self.manager.set_judge_fn(judge)
        block = self.manager._decision_to_memory_block("short")
        assert block is None  # len < 10
        assert call_count == 0  # 未调用 judge_fn

    def test_consolidate_with_empty_judge_default(self) -> None:
        """验证无 judge_fn 时 consolidate 走纯模板路径。"""
        self.manager.set_judge_fn(None)
        summary = "[Compressed]\n## Key Decisions\n- Use Redis: improves performance\n"
        self.manager.consolidate(summary)

        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        assert "Decision: Use Redis" in memory_text

    def test_consolidate_with_section_judge_merges_decisions(self) -> None:
        """验证节级 LLM 评判可以合并多条决策。"""

        def section_judge(section: str) -> list[MemoryJudgeResult]:
            # 模拟 LLM 合并两条决策为一条
            return [
                MemoryJudgeResult(
                    is_worth_remembering=True,
                    confidence=0.9,
                    scope="provider",
                    suggested_title="Decision: Use Redis + streaming",
                    suggested_context="Need caching and async streaming",
                    suggested_solution="Use Redis for caching and streaming protocol",
                    suggested_takeaways="Combined approach improves perf by 60%",
                )
            ]

        self.manager.set_consolidate_judge_fn(section_judge)
        summary = (
            "[Compressed]\n"
            "## Key Decisions\n"
            "- Use Redis for caching\n"
            "- Adopt streaming protocol\n"
        )
        self.manager.consolidate(summary)

        memory_text = self.manager.memory_file.read_text(encoding="utf-8")
        # 两条决策被合并为一条
        assert "Decision: Use Redis + streaming" in memory_text
        assert "Use Redis" in memory_text
        assert "streaming" in memory_text
        trace_events = self.manager.drain_trace_events()
        accepted = [e for e in trace_events if e.type == "accepted"]
        assert len(accepted) == 1  # 只写了一条合并后的记忆

    def test_consolidate_section_judge_fallback_on_error(self) -> None:
        """验证节级评判失败时降级到逐条评判。"""
        calls = []

        def failing_section_judge(section: str) -> list[MemoryJudgeResult]:
            msg = ""
            raise RuntimeError(msg)

        def bullet_judge(text: str) -> MemoryJudgeResult:
            calls.append(text)
            return MemoryJudgeResult(
                is_worth_remembering=True,
                confidence=0.8,
                suggested_title=f"Decision: {text[:40]}",
                suggested_context=text,
                suggested_solution=text,
                suggested_takeaways=text,
            )

        self.manager.set_consolidate_judge_fn(failing_section_judge)
        self.manager.set_judge_fn(bullet_judge)
        summary = "[Compressed]\n## Key Decisions\n- Decision A\n- Decision B\n"
        self.manager.consolidate(summary)

        # 节级失败后降级到逐条评判，两条决策各调用一次 bullet_judge
        assert len(calls) == 2
        assert "Decision A" in calls[0]
        assert "Decision B" in calls[1]

    def test_consolidate_section_judge_rejects_all(self) -> None:
        """验证节级评判可以拒绝所有决策。"""

        def section_judge(section: str) -> list[MemoryJudgeResult]:
            return []

        self.manager.set_consolidate_judge_fn(section_judge)
        summary = "[Compressed]\n## Key Decisions\n- Trivial config change\n"
        self.manager.consolidate(summary)

        assert not self.manager.memory_file.exists()

    def test_record_llm_references_detects_implicit_refs(self) -> None:
        """验证 LLM 引用检测可以捕获隐式引用。"""

        def ref_judge(text: str, candidates: list[str]) -> list[str]:
            # 模拟检测到 "Redis" 相关引用
            for c in candidates:
                if "Redis" in c or "redis" in text.lower():
                    return [c]
            return []

        self.manager.set_reference_judge_fn(ref_judge)

        # 注入一条记忆
        block = (
            "## Decision: Use Redis for caching\n"
            "- Context/Query: Need caching\n"
            "- Solution: Use Redis\n"
            "- Files: config.py\n"
            "- Takeaways: Redis works\n"
        )
        self.manager.add_memory_block(block)

        # 手动设置 _session_usage（模拟注入）
        records = self.manager.read_memory_records()
        for r in records:
            key = (r.layer, r.memory_id)
            from xcode.harness.memory.manager import _SessionMemoryUsage

            self.manager._session_usage[key] = _SessionMemoryUsage(
                retrieved=True, injected=True
            )

        # LLM 引用检测
        count = self.manager.record_llm_references(
            "Used Redis for caching, performance improved"
        )
        assert count == 1

        # 验证标记为 referenced
        for usage in self.manager._session_usage.values():
            assert usage.referenced

    def test_record_llm_references_noop_without_judge(self) -> None:
        """验证未配置 reference_judge_fn 时 record_llm_references 返回 0。"""
        self.manager.set_reference_judge_fn(None)
        count = self.manager.record_llm_references("test")
        assert count == 0

    def test_build_block_from_judge_result(self) -> None:
        """验证 _build_block_from_judge_result 生成正确的记忆块。"""
        result = MemoryJudgeResult(
            is_worth_remembering=True,
            confidence=0.85,
            scope="database",
            related_files=("db.py", "config.py"),
            suggested_title="Decision: Use PostgreSQL",
            suggested_context="Need relational storage",
            suggested_solution="Switch from SQLite to PostgreSQL",
            suggested_takeaways="Better concurrency support",
        )
        block = self.manager._build_block_from_judge_result(result)
        assert block is not None
        assert "Decision: Use PostgreSQL" in block
        assert "- Scope: database" in block
        assert "db.py" in block
        assert "config.py" in block
        assert "- Confidence: 0.85" in block

    def test_build_block_from_judge_result_empty_title(self) -> None:
        """验证空标题时 _build_block_from_judge_result 返回 None。"""
        result = MemoryJudgeResult(
            is_worth_remembering=True,
            confidence=0.5,
            suggested_title="",
        )
        block = self.manager._build_block_from_judge_result(result)
        assert block is None


if __name__ == "__main__":
    pytest.main()
