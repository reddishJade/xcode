"""基于 BM25 的 MEMORY.md 记忆系统。

支持质量门（拒绝低质量/重复块）、冲突合并（同标题合并字段）、
保留策略（超过 max_blocks 时结合质量与使用信号淘汰较弱记忆）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from collections.abc import Callable

from rank_bm25 import BM25Okapi

from .parsing import (
    MemoryEvidence,
    MemoryRecord,
    MemorySearchEvalCase,
    MemorySearchEvalResult,
    MemoryTraceEvent,
    MemoryType,
    _DEFAULT_MAX_BLOCKS,
    _MIN_BLOCK_LENGTH,
    _MIN_FIELD_CONTENT_LENGTH,
    _NOVELTY_THRESHOLD,
    build_memory_id,
    extract_field_content,
    extract_title,
    parse_fields,
    parse_memory_record,
    tokenize,
    tokenize_set,
    with_metadata,
)

type MemoryLayer = Literal["project", "user"]
type MemoryLayerFilter = Literal["all", "project", "user"]
type MemoryOutcome = Literal["success", "failure", "corrected"]

_DEFAULT_SEMANTIC_WEIGHT = 0.4
_EMBEDDING_FAILURES = (ArithmeticError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class MemoryJudgeResult:
    """LLM 记忆质量评判的结构化输出。"""

    is_worth_remembering: bool
    confidence: float = 0.5
    scope: str | None = None
    related_files: tuple[str, ...] = ()
    suggested_title: str = ""
    suggested_context: str = ""
    suggested_solution: str = ""
    suggested_takeaways: str = ""
    reasoning: str = ""


type MemoryJudgeFn = Callable[[str], MemoryJudgeResult]
"""可选的 LLM 记忆质量评估函数。接收一条决策文本，返回结构化评判结果。"""


type MemoryEmbeddingFn = Callable[[str], list[float]]
"""可选的语义嵌入函数。接收文本，返回 float 向量。"""


type MemoryConsolidateJudgeFn = Callable[[str], list[MemoryJudgeResult]]
"""可选的节级 LLM 记忆质量评估函数。

接收完整 Key Decisions 节文本，返回 0+ 条结构化评判结果。
LLM 可以一次看到所有决策的前后文，做合并、筛选、重组。
"""


type MemoryReferenceJudgeFn = Callable[[str, list[str]], list[str]]
"""可选的 LLM 引用检测函数。

接收(文本, 候选记忆标题列表)，返回被引用的记忆标题列表。
用于补充纯子串匹配的不足，检测隐式引用。
"""


@dataclass
class _SessionMemoryUsage:
    retrieved: bool = False
    injected: bool = False
    referenced: bool = False
    adopted: bool = False
    score: float | None = None


@dataclass(frozen=True)
class MemoryRerankPolicy:
    """memory rerank 的显式权重与乘数配置。"""

    lexical_bm25_weight: float = 0.8
    title_weight: float = 1.8
    solution_weight: float = 1.6
    context_weight: float = 1.1
    takeaways_weight: float = 1.0
    file_weight: float = 1.9
    symbol_weight: float = 2.2
    lexical_score_cap: float = 3.5

    exact_file_match_bonus: float = 1.2
    exact_basename_bonus: float = 0.8
    exact_symbol_match_bonus: float = 1.1
    phrase_match_bonus: float = 0.35

    deprecated_status_multiplier: float = 0.2
    confidence_base: float = 0.75
    confidence_scale: float = 0.5
    needs_review_multiplier: float = 0.7
    utility_scale: float = 0.25
    utility_multiplier_min: float = 0.4
    utility_multiplier_max: float = 1.6
    provenance_bonus: float = 1.1
    scope_hit_multiplier: float = 1.35
    scope_mismatch_multiplier: float = 0.75
    freshness_half_life_days: float = 30.0
    freshness_multiplier_min: float = 0.55
    freshness_multiplier_max: float = 1.1
    recent_window_days: float = 7.0
    failed_reuse_penalty: float = 0.35
    corrected_reuse_penalty: float = 0.55
    current_file_bonus: float = 1.2
    recent_file_bonus: float = 0.35
    symbol_context_bonus: float = 1.15
    error_context_bonus: float = 0.75
    task_phase_bonus: float = 0.3
    module_context_bonus: float = 0.4


@dataclass(frozen=True)
class MemoryRetrievalContext:
    """结构化 memory 检索上下文。"""

    query: str
    scope: str | None = None
    current_file: str | None = None
    symbols: tuple[str, ...] = ()
    error_messages: tuple[str, ...] = ()
    task_phase: str | None = None
    modules: tuple[str, ...] = ()
    recent_files: tuple[str, ...] = ()

    def lexical_text(self) -> str:
        parts = [self.query]
        if self.current_file:
            parts.append(self.current_file)
        parts.extend(self.recent_files)
        parts.extend(self.symbols)
        parts.extend(self.error_messages)
        if self.task_phase:
            parts.append(self.task_phase)
        parts.extend(self.modules)
        if self.scope:
            parts.append(self.scope)
        return "\n".join(part.strip() for part in parts if part and part.strip())


class MemoryManager:
    """基于 H2 契约校验、BM25 召回和元数据重排的 MEMORY.md 记忆系统。"""

    def __init__(
        self,
        root: Path,
        max_blocks: int = _DEFAULT_MAX_BLOCKS,
        user_memory_file: Path | None = None,
        min_retrieval_score: float = 0.2,
        min_confidence: float = 0.0,
        rerank_policy: MemoryRerankPolicy | None = None,
        judge_fn: MemoryJudgeFn | None = None,
        embedding_fn: MemoryEmbeddingFn | None = None,
        consolidate_judge_fn: MemoryConsolidateJudgeFn | None = None,
        reference_judge_fn: MemoryReferenceJudgeFn | None = None,
    ) -> None:
        """创建项目级与用户级并行的记忆管理器。

        参数：
            judge_fn: 可选的 LLM 记忆质量评估函数。接收决策文本，返回结构化评判结果。
                      启用后在 consolidation 中使用 LLM 判断替代纯模板抽取。
            embedding_fn: 可选的语义嵌入函数。接收文本返回 float 向量，
                           启用后在检索中结合 BM25 做混合排序。
        """
        self.root = root
        self.memory_file = root / "MEMORY.md"
        self.user_memory_file = user_memory_file or (
            Path.home() / ".xcode" / "memory" / "MEMORY.md"
        )
        self.archive_dir = root / ".local" / "memory_archive"
        self.lru_file = root / ".local" / "memory_lru.json"
        self.max_blocks = max_blocks
        self.min_retrieval_score = min_retrieval_score
        self.min_confidence = min_confidence
        self.rerank_policy = rerank_policy or MemoryRerankPolicy()
        self.judge_fn: MemoryJudgeFn | None = judge_fn
        self.embedding_fn: MemoryEmbeddingFn | None = embedding_fn
        self.consolidate_judge_fn: MemoryConsolidateJudgeFn | None = (
            consolidate_judge_fn
        )
        self.reference_judge_fn: MemoryReferenceJudgeFn | None = reference_judge_fn
        self._trace_events: list[MemoryTraceEvent] = []
        self._session_usage: dict[tuple[str, str], _SessionMemoryUsage] = {}

    # ── 读取 ──

    def read_memory_blocks(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> list[str]:
        """读取指定层级的记忆块；默认合并项目级与用户级。"""
        blocks: list[str] = []
        for current_layer in self._selected_layers(layer):
            blocks.extend(self._read_blocks_from_file(self._memory_file(current_layer)))
        return blocks

    def read_memory_records(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> list[MemoryRecord]:
        """读取指定层级并保留来源信息。"""
        records: list[MemoryRecord] = []
        for current_layer in self._selected_layers(layer):
            memory_file = self._memory_file(current_layer)
            for block in self._read_blocks_from_file(memory_file):
                record = parse_memory_record(block, layer=current_layer)
                records.append(record)
        return records

    # ── 预算控制注入 ──

    def read_budgeted(
        self,
        max_tokens: int,
        layer: MemoryLayerFilter = "all",
    ) -> list[str]:
        """按 token 预算读取记忆块，重要性高的优先。"""
        if max_tokens <= 0:
            return []
        from xcode.agent._compaction import estimate_tokens

        records = self.read_memory_records(layer=layer)
        if not records:
            return []
        sorted_records = self._sort_by_importance(records)
        result: list[str] = []
        budget = max_tokens
        for record in sorted_records:
            packet = self.render_prompt_packet(record)
            tokens = estimate_tokens(packet)
            if tokens > budget and result:
                break
            if tokens <= budget:
                result.append(packet)
                budget -= tokens
        return result

    def read_budgeted_records(
        self,
        max_tokens: int,
        layer: MemoryLayerFilter = "all",
    ) -> list[MemoryRecord]:
        """按 token 预算读取记忆记录，重要性高的优先。"""
        if max_tokens <= 0:
            return []
        from xcode.agent._compaction import estimate_tokens

        records = self.read_memory_records(layer=layer)
        if not records:
            return []
        sorted_records = self._sort_by_importance(records)
        result: list[MemoryRecord] = []
        budget = max_tokens
        for record in sorted_records:
            packet = self.render_prompt_packet(record)
            tokens = estimate_tokens(packet)
            if tokens > budget and result:
                break
            if tokens <= budget:
                result.append(record)
                budget -= tokens
        return result

    def select_budgeted_records(
        self,
        candidates: Sequence[MemoryRecord],
        *,
        max_tokens: int,
    ) -> list[MemoryRecord]:
        """按当前检索顺序装箱，不用全局重要性替换更相关记录。"""
        if max_tokens <= 0:
            return []
        from xcode.agent._compaction import estimate_tokens

        selected: list[MemoryRecord] = []
        remaining = max_tokens
        for record in candidates:
            tokens = estimate_tokens(self.render_prompt_packet(record))
            if tokens > remaining:
                break
            selected.append(record)
            remaining -= tokens
        return selected

    def _sort_by_importance(
        self,
        records: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """按重要性排序：类型 > 状态 > 效用 > 新旧。"""
        type_rank = {
            "semantic": 4,
            "procedural": 3,
            "preference": 2,
            "episodic": 1,
            "": 2,
        }
        status_rank = {
            "active": 3,
            "needs_review": 2,
            "deprecated": 0,
            "superseded": 0,
            "obsolete": 0,
        }

        def rank(r: MemoryRecord) -> tuple[int, int, float, str | None]:
            t = type_rank.get(r.memory_type, 2)
            s = status_rank.get(r.status, 1)
            u = r.utility
            age = r.modified_at or r.created_at or ""
            return (t, s, u, age)

        return sorted(records, key=rank, reverse=True)

    def _read_blocks_from_file(self, memory_file: Path) -> list[str]:
        """从单个 MEMORY.md 文件解析 H2 记忆块。"""
        if not memory_file.exists():
            return []
        content = memory_file.read_text(encoding="utf-8")
        blocks: list[str] = []
        current_block: list[str] = []
        for line in content.splitlines():
            if line.startswith("## "):
                if current_block:
                    blocks.append("\n".join(current_block) + "\n")
                current_block = [line]
            else:
                if current_block or line.strip():
                    current_block.append(line)
        if current_block:
            blocks.append("\n".join(current_block) + "\n")
        return [block for block in blocks if block.strip()]

    # ── 检索 ──

    def search_memory(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
        layer: MemoryLayerFilter = "all",
        retrieval_context: MemoryRetrievalContext | None = None,
    ) -> list[str]:
        """跨项目级与用户级记忆检索匹配块。"""
        records = self.search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=layer,
            retrieval_context=retrieval_context,
        )
        return [record.block for record in records]

    def search_memory_records(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
        layer: MemoryLayerFilter = "all",
        *,
        source: str = "api",
        track_usage: bool = True,
        retrieval_context: MemoryRetrievalContext | None = None,
    ) -> list[MemoryRecord]:
        """跨层级执行统一的混合检索、重排和 gate。"""
        return self._search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=layer,
            source=source,
            track_usage=track_usage,
            retrieval_context=retrieval_context,
            semantic_weight=_DEFAULT_SEMANTIC_WEIGHT,
        )

    def _search_memory_records(
        self,
        query: str,
        *,
        limit: int,
        scope: str | None,
        layer: MemoryLayerFilter,
        source: str,
        track_usage: bool,
        retrieval_context: MemoryRetrievalContext | None,
        semantic_weight: float,
    ) -> list[MemoryRecord]:
        """执行唯一的检索流水线，供公开入口和兼容入口复用。"""
        started_at = time.perf_counter()
        context = self._coerce_retrieval_context(
            query,
            scope=scope,
            retrieval_context=retrieval_context,
        )
        lexical_candidates = self.retrieve_memory_candidates(
            context,
            layer=layer,
        )
        if not lexical_candidates or limit <= 0:
            if source == "tool":
                self._emit_trace(
                    MemoryTraceEvent(
                        type="tool_searched",
                        latency_ms=self._elapsed_ms(started_at),
                        source=source,
                    )
                )
            return []

        candidates = self.fuse_memory_candidates(
            lexical_candidates,
            context,
            semantic_weight=semantic_weight,
        )

        ranked = self.rerank_memory_candidates(
            candidates,
            context,
            scope=scope,
            limit=limit,
        )
        elapsed_ms = self._elapsed_ms(started_at)
        if track_usage and ranked:
            self._touch_lru(ranked)
            self._mark_session_usage(ranked, usage="retrieved")
        for record in ranked:
            self._emit_trace(
                MemoryTraceEvent(
                    type="retrieved",
                    memory_id=self._memory_id(record.layer, record.title),
                    layer=record.layer,
                    title=record.title,
                    score=record.score,
                    latency_ms=elapsed_ms,
                    source=source,
                )
            )
        if source == "tool":
            self._emit_trace(
                MemoryTraceEvent(
                    type="tool_searched",
                    latency_ms=elapsed_ms,
                    source=source,
                )
            )
        return ranked

    def retrieve_memory_candidates(
        self,
        query: str | MemoryRetrievalContext,
        *,
        layer: MemoryLayerFilter = "all",
    ) -> list[MemoryRecord]:
        """执行 BM25 词法召回，返回带词法分数的候选。"""
        context = self._coerce_retrieval_context(query)
        records = self.read_memory_records(layer=layer)
        blocks = [record.block for record in records]
        lexical_query = context.lexical_text()
        if not blocks or not lexical_query.strip():
            return []

        corpus = [tokenize(block) for block in blocks]
        query_words = tokenize(lexical_query)
        bm25 = BM25Okapi(corpus)
        raw = list(bm25.get_scores(query_words))
        if not raw:
            scores = []
        elif max(raw) - min(raw) > 1e-6:
            lo, hi = min(raw), max(raw)
            scores = [(s - lo) / (hi - lo) for s in raw]
        else:
            query_set = set(query_words)
            scores = [
                sum(q in b for q in query_set) / max(len(query_words), 1)
                for b in corpus
            ]

        candidates: list[MemoryRecord] = []
        for score, record in zip(scores, records, strict=True):
            lexical = self._weighted_lexical_score(record, context, bm25_score=score)
            candidates.append(replace(record, score=round(lexical, 6)))
        candidates.sort(key=lambda r: (-r.score, r.title))
        return candidates

    def fuse_memory_candidates(
        self,
        candidates: Sequence[MemoryRecord],
        query: str | MemoryRetrievalContext,
        *,
        semantic_weight: float = _DEFAULT_SEMANTIC_WEIGHT,
    ) -> list[MemoryRecord]:
        """融合词法与语义分数；嵌入不可用时原样返回词法候选。"""
        context = self._coerce_retrieval_context(query)
        semantic_scores = self._semantic_scores(candidates, context)
        if semantic_scores is None:
            return list(candidates)

        bounded_weight = min(max(semantic_weight, 0.0), 1.0)
        lexical_max = max((record.score for record in candidates), default=0.0)
        semantic_max = max(semantic_scores, default=0.0)
        score_scale = max(lexical_max, 1.0)
        fused: list[MemoryRecord] = []
        for record, semantic_score in zip(candidates, semantic_scores, strict=True):
            lexical_norm = record.score / lexical_max if lexical_max > 0 else 0.0
            semantic_norm = (
                max(semantic_score, 0.0) / semantic_max if semantic_max > 0 else 0.0
            )
            score = (
                lexical_norm * (1.0 - bounded_weight) + semantic_norm * bounded_weight
            ) * score_scale
            fused.append(replace(record, score=round(score, 6)))
        fused.sort(key=lambda record: (-record.score, record.title))
        return fused

    def _semantic_scores(
        self,
        candidates: Sequence[MemoryRecord],
        context: MemoryRetrievalContext,
    ) -> list[float] | None:
        """计算语义相似度；嵌入失败时返回 None 触发词法降级。"""
        if self.embedding_fn is None:
            return None
        try:
            query_vector = self.embedding_fn(context.lexical_text())
            if not query_vector or not all(
                math.isfinite(value) for value in query_vector
            ):
                raise ValueError("查询嵌入向量无效")
            scores: list[float] = []
            for record in candidates:
                record_vector = self.embedding_fn(record.block)
                if len(record_vector) != len(query_vector) or not all(
                    math.isfinite(value) for value in record_vector
                ):
                    raise ValueError("记忆嵌入向量无效")
                scores.append(_cosine_similarity(query_vector, record_vector))
            return scores
        except _EMBEDDING_FAILURES:
            return None

    def rerank_memory_candidates(
        self,
        candidates: Sequence[MemoryRecord],
        query: str | MemoryRetrievalContext,
        *,
        scope: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """对 lexical candidates 应用可替换的 rerank 与 gate。"""
        context = self._coerce_retrieval_context(query, scope=scope)
        ranked: list[MemoryRecord] = []
        for candidate in candidates:
            adjusted = self._apply_rerank_policy(
                candidate,
                context.query,
                context.scope,
            )
            if not self._passes_retrieval_gate(candidate, adjusted):
                continue
            ranked.append(replace(candidate, score=round(adjusted, 6)))
        ranked.sort(key=lambda r: (-r.score, r.title))
        if limit is not None and limit > 0:
            return ranked[:limit]
        return ranked

    def _passes_retrieval_gate(
        self,
        candidate: MemoryRecord,
        score: float,
    ) -> bool:
        """统一执行最终分数阈值与置信度 gate。"""
        return (
            math.isfinite(score)
            and score >= self.min_retrieval_score
            and self._passes_confidence_gate(candidate)
        )

    # ── 评测 ──

    def evaluate_search(
        self,
        cases: list[MemorySearchEvalCase],
        limit: int = 3,
    ) -> list[MemorySearchEvalResult]:
        results: list[MemorySearchEvalResult] = []
        for case in cases:
            records = self.search_memory_records(
                case.query, limit=limit, scope=case.scope
            )
            titles = tuple(r.title for r in records)
            expected = case.expected_title_contains
            excluded_hits = tuple(
                title
                for title in titles
                if any(blocked in title for blocked in case.excluded_title_contains)
            )
            results.append(
                MemorySearchEvalResult(
                    query=case.query,
                    passed=any(expected in title for title in titles)
                    and not excluded_hits,
                    expected_title_contains=expected,
                    matched_titles=titles,
                    excluded_title_hits=excluded_hits,
                )
            )
        return results

    # ── 校验 ──

    def validate_memory_block(self, block: str) -> bool:
        content = block.strip()
        if not content.startswith("## "):
            return False
        for field in ["Context/Query", "Solution", "Files", "Takeaways"]:
            if field not in content:
                return False
        return True

    def _content_quality_check(self, block: str) -> bool:
        content = block.strip()
        if len(content) < _MIN_BLOCK_LENGTH:
            return False
        for field_name in ("Context/Query", "Solution", "Files", "Takeaways"):
            field_value = extract_field_content(content, field_name)
            if (
                field_value is None
                or len(field_value.strip()) < _MIN_FIELD_CONTENT_LENGTH
            ):
                return False
        return True

    def _quality_check(
        self, block: str, existing_records: list[MemoryRecord] | None = None
    ) -> bool:
        return self._quality_rejection_reason(block, existing_records) is None

    def _weighted_lexical_score(
        self,
        record: MemoryRecord,
        context: MemoryRetrievalContext,
        *,
        bm25_score: float,
    ) -> float:
        query_terms = tokenize_set(context.lexical_text())
        if not query_terms:
            return 0.0
        score = bm25_score * self.rerank_policy.lexical_bm25_weight
        score += self._field_overlap_score(
            query_terms,
            record.title,
            weight=self.rerank_policy.title_weight,
        )
        score += self._field_overlap_score(
            query_terms,
            record.fields.get("solution", ""),
            weight=self.rerank_policy.solution_weight,
        )
        score += self._field_overlap_score(
            query_terms,
            record.fields.get("context/query", ""),
            weight=self.rerank_policy.context_weight,
        )
        score += self._field_overlap_score(
            query_terms,
            record.fields.get("takeaways", ""),
            weight=self.rerank_policy.takeaways_weight,
        )
        score += self._field_overlap_score(
            query_terms,
            ", ".join(record.related_files or ()) or record.fields.get("files", ""),
            weight=self.rerank_policy.file_weight,
        )
        score += self._field_overlap_score(
            query_terms,
            ", ".join(record.related_symbols),
            weight=self.rerank_policy.symbol_weight,
        )
        score += self._exact_match_bonus(record, context.query)
        score += self._structured_context_bonus(record, context)
        return min(score, self.rerank_policy.lexical_score_cap)

    def _field_overlap_score(
        self,
        query_terms: set[str],
        text: str,
        *,
        weight: float,
    ) -> float:
        field_terms = tokenize_set(text)
        if not field_terms:
            return 0.0
        overlap = len(query_terms & field_terms) / max(len(query_terms), 1)
        return overlap * weight

    def _apply_rerank_policy(
        self,
        candidate: MemoryRecord,
        query: str,
        scope: str | None,
    ) -> float:
        adjusted = candidate.score
        if adjusted <= 0:
            return 0.0
        if candidate.status in {"deprecated", "superseded", "obsolete"}:
            adjusted *= self.rerank_policy.deprecated_status_multiplier
        confidence = candidate.confidence_value
        if confidence is not None:
            bounded = min(max(confidence, 0.0), 1.0)
            adjusted *= (
                self.rerank_policy.confidence_base
                + bounded * self.rerank_policy.confidence_scale
            )
        if candidate.status == "needs_review":
            adjusted *= self.rerank_policy.needs_review_multiplier
        if candidate.utility != 0.0:
            adjusted *= max(
                self.rerank_policy.utility_multiplier_min,
                min(
                    self.rerank_policy.utility_multiplier_max,
                    1.0 + candidate.utility * self.rerank_policy.utility_scale,
                ),
            )
        adjusted *= self._negative_transfer_multiplier(candidate, query, scope)
        if scope:
            adjusted *= self._scope_multiplier(candidate, scope)
        adjusted *= self._freshness_multiplier(candidate)
        query_terms = set(tokenize(query))
        provenance_text = " ".join(
            candidate.fields.get(key, "")
            for key in ("source", "session", "validated", "validation")
        )
        if query_terms and query_terms.intersection(tokenize(provenance_text)):
            adjusted *= self.rerank_policy.provenance_bonus
        return adjusted

    def _scope_multiplier(self, candidate: MemoryRecord, scope: str) -> float:
        scope_terms = set(tokenize(scope))
        if not scope_terms:
            return 1.0
        scoped_text = " ".join(
            candidate.fields.get(key, "")
            for key in ("scope", "files", "context/query", "takeaways")
        )
        scoped_terms = set(tokenize(scoped_text))
        if scope_terms.intersection(scoped_terms):
            return self.rerank_policy.scope_hit_multiplier
        if candidate.fields.get("scope"):
            return self.rerank_policy.scope_mismatch_multiplier
        return 1.0

    def _freshness_multiplier(self, record: MemoryRecord) -> float:
        timestamp = self._record_timestamp(record)
        if timestamp is None:
            return 1.0
        age_days = max((time.time() - timestamp) / 86400.0, 0.0)
        half_life_days = max(self.rerank_policy.freshness_half_life_days, 1.0)
        recent_window_days = max(self.rerank_policy.recent_window_days, 0.0)
        max_multiplier = max(self.rerank_policy.freshness_multiplier_max, 1.0)
        min_multiplier = min(self.rerank_policy.freshness_multiplier_min, 1.0)
        if age_days <= recent_window_days:
            if recent_window_days == 0:
                return max_multiplier
            boost_ratio = 1.0 - age_days / recent_window_days
            return 1.0 + (max_multiplier - 1.0) * boost_ratio
        decay = 0.5 ** ((age_days - recent_window_days) / half_life_days)
        return max(min_multiplier, decay)

    def _record_timestamp(self, record: MemoryRecord) -> float | None:
        for value in (record.modified_at, record.created_at):
            parsed = self._parse_timestamp(value)
            if parsed is not None:
                return parsed
        return self.get_last_used_at(record)

    def _parse_timestamp(self, value: str | None) -> float | None:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    def _negative_transfer_multiplier(
        self,
        record: MemoryRecord,
        query: str,
        scope: str | None,
    ) -> float:
        if self._has_reuse_boundary_match(record, query, scope):
            return 1.0
        if (
            record.failure_count > record.success_count
            or record.last_outcome == "failure"
        ):
            return self.rerank_policy.failed_reuse_penalty
        if record.correction_count > 0 or record.last_outcome == "corrected":
            return self.rerank_policy.corrected_reuse_penalty
        return 1.0

    def _has_reuse_boundary_match(
        self,
        record: MemoryRecord,
        query: str,
        scope: str | None,
    ) -> bool:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return False
        related_files = tuple(item.lower() for item in record.related_files)
        if normalized_query in related_files:
            return True
        if normalized_query in {Path(item).name.lower() for item in related_files}:
            return True
        related_symbols = {symbol.lower() for symbol in record.related_symbols}
        if normalized_query in related_symbols:
            return True
        if scope:
            scope_terms = set(tokenize(scope))
            if scope_terms:
                record_scope_terms = tokenize_set(record.fields.get("scope", ""))
                if scope_terms.intersection(record_scope_terms):
                    return True
        return False

    def _exact_match_bonus(self, record: MemoryRecord, query: str) -> float:
        bonus = 0.0
        normalized_query = query.strip().lower()
        if not normalized_query:
            return 0.0
        related_files = record.related_files or ()
        if any(normalized_query == item.lower() for item in related_files):
            bonus += self.rerank_policy.exact_file_match_bonus
        elif normalized_query in {Path(item).name.lower() for item in related_files}:
            bonus += self.rerank_policy.exact_basename_bonus
        related_symbols = {symbol.lower() for symbol in record.related_symbols}
        if normalized_query in related_symbols:
            bonus += self.rerank_policy.exact_symbol_match_bonus
        phrase_fields = [
            record.title.lower(),
            record.fields.get("context/query", "").lower(),
            record.fields.get("solution", "").lower(),
        ]
        if len(tokenize(query)) >= 2 and any(
            normalized_query in field for field in phrase_fields
        ):
            bonus += self.rerank_policy.phrase_match_bonus
        return bonus

    def _structured_context_bonus(
        self,
        record: MemoryRecord,
        context: MemoryRetrievalContext,
    ) -> float:
        bonus = 0.0
        related_files = tuple(item.lower() for item in record.related_files)
        if context.current_file:
            current_file = context.current_file.strip().lower()
            if current_file in related_files:
                bonus += self.rerank_policy.current_file_bonus
            elif current_file in {Path(item).name.lower() for item in related_files}:
                bonus += self.rerank_policy.current_file_bonus * 0.7
        if context.recent_files:
            recent_files = {
                item.strip().lower() for item in context.recent_files if item.strip()
            }
            if recent_files.intersection(related_files):
                bonus += self.rerank_policy.recent_file_bonus
        related_symbols = {symbol.lower() for symbol in record.related_symbols}
        if related_symbols and {
            symbol.strip().lower() for symbol in context.symbols if symbol.strip()
        }.intersection(related_symbols):
            bonus += self.rerank_policy.symbol_context_bonus
        if context.error_messages:
            phrase_fields = (
                record.fields.get("context/query", ""),
                record.fields.get("solution", ""),
                record.fields.get("takeaways", ""),
            )
            for message in context.error_messages:
                normalized = message.strip().lower()
                if not normalized:
                    continue
                if any(normalized in field.lower() for field in phrase_fields):
                    bonus += self.rerank_policy.error_context_bonus
                    break
        if context.task_phase:
            phase_terms = tokenize_set(context.task_phase)
            if phase_terms:
                phase_text = " ".join(
                    (
                        record.fields.get("context/query", ""),
                        record.fields.get("takeaways", ""),
                    )
                )
                if phase_terms.intersection(tokenize_set(phase_text)):
                    bonus += self.rerank_policy.task_phase_bonus
        if context.modules:
            module_terms = {
                module.strip().lower() for module in context.modules if module.strip()
            }
            scope_text = " ".join(
                (
                    record.fields.get("scope", ""),
                    record.fields.get("files", ""),
                    ", ".join(record.related_files),
                )
            ).lower()
            if any(module in scope_text for module in module_terms):
                bonus += self.rerank_policy.module_context_bonus
        return bonus

    def _coerce_retrieval_context(
        self,
        query: str | MemoryRetrievalContext,
        *,
        scope: str | None = None,
        retrieval_context: MemoryRetrievalContext | None = None,
    ) -> MemoryRetrievalContext:
        if retrieval_context is not None:
            return retrieval_context
        if isinstance(query, MemoryRetrievalContext):
            if scope is None or query.scope == scope:
                return query
            return MemoryRetrievalContext(
                query=query.query,
                scope=scope,
                current_file=query.current_file,
                symbols=query.symbols,
                error_messages=query.error_messages,
                task_phase=query.task_phase,
                modules=query.modules,
                recent_files=query.recent_files,
            )
        return MemoryRetrievalContext(query=query, scope=scope)

    def _quality_rejection_reason(
        self,
        block: str,
        existing_records: list[MemoryRecord] | None = None,
    ) -> str | None:
        if not self._content_quality_check(block):
            return "content_quality_failed"
        if existing_records and len(existing_records) > 0:
            if self._is_duplicate(block, existing_records):
                return "duplicate_block"
        return None

    def _is_duplicate(self, block: str, existing_records: list[MemoryRecord]) -> bool:
        new_tokens = tokenize_set(block)
        if not new_tokens:
            return False
        for record in existing_records:
            old_tokens = tokenize_set(record.block)
            if not old_tokens:
                continue
            overlap = len(new_tokens & old_tokens) / min(
                len(new_tokens), len(old_tokens)
            )
            if overlap >= _NOVELTY_THRESHOLD:
                return True
        return False

    # ── 合并与写入 ──

    def add_memory_block(
        self,
        block: str,
        *,
        source: str | None = None,
        scope: str | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        status: str | None = None,
        validity: str | None = None,
        supersedes: Sequence[str] = (),
        evidence: Sequence[MemoryEvidence] = (),
        layer: MemoryLayer = "project",
    ) -> bool:
        """校验并写入指定记忆层级。"""
        return self._persist_memory_block(
            block,
            source=source,
            scope=scope,
            confidence=confidence,
            memory_type=memory_type,
            status=status,
            validity=validity,
            supersedes=tuple(supersedes),
            evidence=evidence,
            retrieval_count=None,
            injection_count=None,
            reference_count=None,
            adoption_count=None,
            success_count=None,
            failure_count=None,
            correction_count=None,
            utility=None,
            last_outcome=None,
            layer=layer,
            emit_candidate_trace=True,
        )

    def _persist_memory_block(
        self,
        block: str,
        *,
        source: str | None,
        scope: str | None,
        confidence: float | None,
        memory_type: MemoryType | None,
        status: str | None,
        validity: str | None,
        supersedes: Sequence[str],
        evidence: Sequence[MemoryEvidence],
        retrieval_count: int | None,
        injection_count: int | None,
        reference_count: int | None,
        adoption_count: int | None,
        success_count: int | None,
        failure_count: int | None,
        correction_count: int | None,
        utility: float | None,
        last_outcome: str | None,
        layer: MemoryLayer,
        emit_candidate_trace: bool,
    ) -> bool:
        """执行正式记忆写入与合并。"""
        title = extract_title(block)
        if emit_candidate_trace:
            self._emit_candidate_trace(title, layer, source)

        if not self._reject_invalid_candidate(block, layer, source, title):
            return False

        block = with_metadata(
            block,
            layer=layer,
            source=source,
            scope=scope,
            confidence=confidence,
            memory_type=memory_type,
            status=status,
            validity=validity,
            supersedes=tuple(supersedes),
            evidence=tuple(evidence),
            retrieval_count=retrieval_count,
            injection_count=injection_count,
            reference_count=reference_count,
            adoption_count=adoption_count,
            success_count=success_count,
            failure_count=failure_count,
            correction_count=correction_count,
            utility=utility,
            last_outcome=last_outcome,
        )

        existing_records = self.read_memory_records(layer=layer)
        new_title = extract_title(block)

        if new_title and existing_records:
            if self._try_merge_block(block, new_title, existing_records, layer, source):
                return True

        if self._reject_if_low_quality(
            block, new_title, existing_records, layer, source
        ):
            return False

        return self._write_new_block(block, new_title, layer, source)

    def _emit_candidate_trace(
        self,
        title: str,
        layer: MemoryLayer,
        source: str | None,
    ) -> None:
        """发出候选块创建追踪事件。"""
        self._emit_trace(
            MemoryTraceEvent(
                type="candidate_created",
                memory_id=self._memory_id(layer, title) if title else None,
                layer=layer,
                title=title or None,
                source=source,
            )
        )

    def _reject_invalid_candidate(
        self,
        block: str,
        layer: MemoryLayer,
        source: str | None,
        title: str,
    ) -> bool:
        """校验记忆块 schema，无效则归档并返回 False。"""
        if self.validate_memory_block(block):
            return True
        self._emit_trace(
            MemoryTraceEvent(
                type="rejected",
                memory_id=self._memory_id(layer, title) if title else None,
                layer=layer,
                title=title or None,
                rejection_reason="schema_validation_failed",
                source=source,
            )
        )
        self._archive_block(block, layer)
        return False

    def _try_merge_block(
        self,
        block: str,
        new_title: str,
        existing_records: list[MemoryRecord],
        layer: MemoryLayer,
        source: str | None,
    ) -> bool:
        """尝试与已有记录合并，合并成功并通过质量门则替换并返回 True。"""
        merged_block = self._merge_with_existing(block, new_title, existing_records)
        if merged_block is None:
            return False
        if not self._content_quality_check(merged_block):
            self._emit_trace(
                MemoryTraceEvent(
                    type="rejected",
                    memory_id=self._memory_id(layer, new_title),
                    layer=layer,
                    title=new_title,
                    rejection_reason="merged_quality_gate_failed",
                    source=source,
                )
            )
            self._archive_block(merged_block, layer)
            return False
        self._emit_trace(
            MemoryTraceEvent(
                type="superseded",
                memory_id=self._memory_id(layer, new_title),
                layer=layer,
                title=new_title,
                source=source,
            )
        )
        self._replace_block_by_title(new_title, merged_block, layer)
        self._enforce_lru(layer)
        self._emit_trace(
            MemoryTraceEvent(
                type="accepted",
                memory_id=self._memory_id(layer, new_title),
                layer=layer,
                title=new_title,
                source=source,
            )
        )
        return True

    def _reject_if_low_quality(
        self,
        block: str,
        new_title: str,
        existing_records: list[MemoryRecord],
        layer: MemoryLayer,
        source: str | None,
    ) -> bool:
        """质量门检查，拒绝则归档并返回 True。"""
        rejection_reason = self._quality_rejection_reason(block, existing_records)
        if rejection_reason is None:
            return False
        self._emit_trace(
            MemoryTraceEvent(
                type="rejected",
                memory_id=self._memory_id(layer, new_title) if new_title else None,
                layer=layer,
                title=new_title or None,
                rejection_reason=rejection_reason,
                source=source,
            )
        )
        self._archive_block(block, layer)
        return True

    def _write_new_block(
        self,
        block: str,
        new_title: str,
        layer: MemoryLayer,
        source: str | None,
    ) -> bool:
        """将记忆块追加到存储文件。"""
        memory_file = self._memory_file(layer)
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        with memory_file.open("a", encoding="utf-8") as file:
            file.write("\n" + block.strip() + "\n")
        self._enforce_lru(layer)
        self._emit_trace(
            MemoryTraceEvent(
                type="accepted",
                memory_id=self._memory_id(layer, new_title) if new_title else None,
                layer=layer,
                title=new_title or None,
                source=source,
            )
        )
        return True

    def _replace_block_by_title(
        self,
        title: str,
        new_block: str,
        layer: MemoryLayer,
    ) -> None:
        """按标题替换指定层级中的现有记忆块。"""
        blocks = self.read_memory_blocks(layer=layer)
        updated: list[str] = []
        found = False
        for existing_block in blocks:
            existing_title = extract_title(existing_block)
            if existing_title and existing_title.lower() == title.lower():
                updated.append(new_block.strip() + "\n")
                found = True
            else:
                updated.append(existing_block.rstrip() + "\n")
        if not found:
            updated.append(new_block.strip() + "\n")
        self._memory_file(layer).write_text("".join(updated), encoding="utf-8")

    def _merge_with_existing(
        self,
        new_block: str,
        new_title: str,
        existing_records: list[MemoryRecord],
    ) -> str | None:
        new_lower = new_title.lower()
        for record in existing_records:
            if record.title.lower() == new_lower:
                new_fields = parse_fields(new_block)
                old_fields = record.fields
                merged = {}
                merged.update(old_fields)
                for key, value in new_fields.items():
                    if value.strip():
                        merged[key] = value
                merged["last_modified"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime()
                )

                lines = [f"## {new_title}"]
                mandatory = ["Context/Query", "Solution", "Files", "Takeaways"]
                for field in mandatory:
                    key = field.lower()
                    if key in merged:
                        lines.append(f"- {field}: {merged.pop(key)}")
                for key, value in merged.items():
                    display_key = key.capitalize() if "_" not in key else key
                    lines.append(f"- {display_key}: {value}")
                return "\n".join(lines) + "\n"
        return None

    # ── 摘要处理（Goal/Progress/Key Decisions/Next Steps）──

    @staticmethod
    def _parse_structured_summary(
        summary: str,
    ) -> dict[str, str]:
        """将结构化压缩摘要解析为 sections。

        返回格式：
        {"goal": "...", "constraints & preferences": "...", "progress": "...",
         "key decisions": "...", "next steps": "...", "critical context": "..."}
        """
        content = summary.removeprefix("[Compressed]").strip()
        sections: dict[str, str] = {}
        current_key = None
        current_lines: list[str] = []
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("## "):
                if current_key is not None:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = stripped[3:].strip().lower()
                current_lines = []
            elif current_key is not None:
                current_lines.append(stripped)
        if current_key is not None:
            sections[current_key] = "\n".join(current_lines).strip()
        return sections

    def consolidate(self, summary: str) -> None:
        """从压缩摘要中提取可复用记忆并写入正式存储。

        解析 Goal/Progress/Key Decisions/Next Steps 结构化格式的摘要，
        从 Key Decisions 节提取架构决策记忆，从 Goal 节提取项目上下文（首次出现时）。

        当配置了 consolidate_judge_fn 时，将整个 Key Decisions 节文本交给 LLM，
        由 LLM 一次完成筛选、合并、重组；否则逐条子弹走 judge_fn 或纯模板回退。
        """
        sections = self._parse_structured_summary(summary)

        # Key Decisions：优先使用节级 LLM 评判
        decisions_text = sections.get("key decisions", "").strip()
        if decisions_text:
            if self.consolidate_judge_fn is not None:
                self._consolidate_with_section_judge(decisions_text)
            else:
                self._consolidate_with_bullet_judge(decisions_text)

        # Goal → project context（仅当文件为空时）
        goal = sections.get("goal", "").strip()
        if goal and len(goal) > 30 and self._should_seed_project_context(goal):
            block = (
                f"## Project context\n"
                f"- Context/Query: {goal.split(chr(10))[0] if chr(10) in goal else goal}\n"
                f"- Solution: (learn from ongoing work)\n"
                f"- Files: (see project)\n"
                f"- Takeaways: {goal[:200]}"
            )
            self._ingest_consolidation_candidate(
                block,
                source="consolidation",
                layer="project",
            )

    def _consolidate_with_section_judge(self, decisions_text: str) -> None:
        """使用节级 LLM 评判整个 Key Decisions 节。

        LLM 一次看到所有决策的前后文，可以合并相关条目、
        过滤低价值决策、重组表述。
        """
        try:
            results = self.consolidate_judge_fn(decisions_text)  # type: ignore[misc]
        except Exception:
            # 节级评判失败时降级到逐条评判
            self._consolidate_with_bullet_judge(decisions_text)
            return

        for result in results:
            if not result.is_worth_remembering:
                self._emit_trace(
                    MemoryTraceEvent(
                        type="rejected",
                        memory_id=None,
                        layer="project",
                        title="",
                        rejection_reason="llm_judge_rejected",
                        source="consolidation",
                    )
                )
                continue
            block = self._build_block_from_judge_result(result)
            if block is not None:
                self._ingest_consolidation_candidate(
                    block,
                    source="consolidation",
                    layer="project",
                )

    def _consolidate_with_bullet_judge(self, decisions_text: str) -> None:
        """逐条子弹评判（回退路径：无 consolidate_judge_fn 时使用）。"""
        decisions = self._extract_bullet_items(decisions_text)
        for decision_text in decisions:
            block = self._decision_to_memory_block(decision_text)
            if block is not None:
                self._ingest_consolidation_candidate(
                    block,
                    source="consolidation",
                    layer="project",
                )

    def _build_block_from_judge_result(self, result: MemoryJudgeResult) -> str | None:
        """从 MemoryJudgeResult 构建记忆块。"""
        if not result.suggested_title:
            return None
        title = result.suggested_title
        context = result.suggested_context or ""
        solution = result.suggested_solution or ""
        takeaways = result.suggested_takeaways or ""
        files = (
            ", ".join(result.related_files) if result.related_files else "(see project)"
        )
        lines = [
            f"## {title}",
            f"- Context/Query: {context}",
            f"- Solution: {solution}",
            f"- Files: {files}",
            f"- Takeaways: {takeaways}",
            f"- Confidence: {result.confidence:.2f}",
            "- status: active",
            "- validity: derived",
            "- memory-type: semantic",
        ]
        if result.scope:
            lines.append(f"- Scope: {result.scope}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_bullet_items(text: str) -> list[str]:
        """从 markdown 文本中提取列表项。"""
        items: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                item = line[2:].strip()
                if item:
                    items.append(item)
            elif line.startswith("* "):
                item = line[2:].strip()
                if item:
                    items.append(item)
            elif (
                line.startswith("1. ")
                or line.startswith("2. ")
                or line.startswith("3. ")
            ):
                item = line[3:].strip()
                if item:
                    items.append(item)
        return items

    def set_judge_fn(self, judge_fn: MemoryJudgeFn | None) -> None:
        """运行时设置 LLM 记忆质量评判函数。"""
        self.judge_fn = judge_fn
        # 未单独设置 consolidate_judge_fn 时同步
        if self.consolidate_judge_fn is None:
            self.consolidate_judge_fn = None

    def set_embedding_fn(self, embedding_fn: MemoryEmbeddingFn | None) -> None:
        """运行时设置语义嵌入函数。"""
        self.embedding_fn = embedding_fn

    def set_consolidate_judge_fn(
        self, consolidate_judge_fn: MemoryConsolidateJudgeFn | None
    ) -> None:
        """运行时设置节级 LLM 记忆质量评判函数。"""
        self.consolidate_judge_fn = consolidate_judge_fn

    def set_reference_judge_fn(
        self, reference_judge_fn: MemoryReferenceJudgeFn | None
    ) -> None:
        """运行时设置 LLM 引用检测函数。"""
        self.reference_judge_fn = reference_judge_fn

    def _decision_to_memory_block(self, decision_text: str) -> str | None:
        """将一条决策文本转换为 MEMORY.md 兼容的记忆块。

        当配置了 judge_fn 时，使用 LLM 判断该决策是否值得记录
        并生成结构化内容；否则回退到纯模板提取。
        """
        decision_text = decision_text.strip()
        if not decision_text or len(decision_text) < 10:
            return None

        if self.judge_fn is not None:
            return self._llm_decision_to_memory_block(decision_text)

        return self._template_decision_to_memory_block(decision_text)

    def _llm_decision_to_memory_block(self, decision_text: str) -> str | None:
        """使用 LLM judge_fn 评判决策并生成结构化记忆块。"""
        result = self.judge_fn(decision_text)  # type: ignore[misc]
        if not result.is_worth_remembering:
            self._emit_trace(
                MemoryTraceEvent(
                    type="rejected",
                    memory_id=None,
                    layer="project",
                    title="",
                    rejection_reason="llm_judge_rejected",
                    source="consolidation",
                )
            )
            return None

        # 使用 LLM 的结构化输出构建记忆块
        title = result.suggested_title or f"Decision: {decision_text[:60]}"
        context = result.suggested_context or decision_text[:200]
        solution = result.suggested_solution or decision_text[:200]
        takeaways = result.suggested_takeaways or decision_text[:200]
        files = (
            ", ".join(result.related_files) if result.related_files else "(see project)"
        )
        scope_line = f"\n- Scope: {result.scope}" if result.scope else ""
        return (
            f"## {title}\n"
            f"- Context/Query: {context}\n"
            f"- Solution: {solution}\n"
            f"- Files: {files}\n"
            f"- Takeaways: {takeaways}\n"
            f"- Confidence: {result.confidence:.2f}\n"
            f"- status: active\n"
            f"- validity: derived\n"
            f"- memory-type: semantic"
            f"{scope_line}\n"
        )

    def _template_decision_to_memory_block(self, decision_text: str) -> str | None:
        """纯模板方式将决策文本转换为记忆块。

        作为 _decision_to_memory_block 的 fallback 路径，
        在未配置 judge_fn 时使用。
        """
        # 去除 **加粗** 标记
        clean = decision_text.replace("**", "").strip()
        # 按冒号分割决策标题与内容
        if ": " in clean:
            title_part, _, body_part = clean.partition(": ")
            title = f"Decision: {title_part.strip()}"
        else:
            title = f"Decision: {clean[:60].rstrip(':').strip()}"
            body_part = clean
        return (
            f"## {title}\n"
            f"- Context/Query: {body_part[:200]}\n"
            f"- Solution: {body_part[:200]}\n"
            f"- Files: (see project)\n"
            f"- Takeaways: {body_part[:200]}\n"
            f"- status: active\n"
            f"- validity: derived\n"
            f"- memory-type: semantic\n"
        )

    def _should_seed_project_context(self, goal: str) -> bool:
        """判断是否应该将目标写入 project context。

        仅在 MEMORY.md 尚无 Project context 块时写入。
        """
        records = self.read_memory_records(layer="project")
        return not any(r.title.lower().startswith("project context") for r in records)

    def _ingest_consolidation_candidate(
        self,
        block: str,
        *,
        source: str,
        layer: MemoryLayer,
    ) -> None:
        """将 compaction 产物经轻量 gate 后直接写入正式记忆。"""
        title = extract_title(block)
        if not self._has_reusable_scope(block):
            self._emit_trace(
                MemoryTraceEvent(
                    type="rejected",
                    memory_id=self._memory_id(layer, title) if title else None,
                    layer=layer,
                    title=title or None,
                    rejection_reason="scope_gate_failed",
                    source=source,
                )
            )
            self._archive_block(block, layer)
            return
        self._persist_memory_block(
            block,
            source=source,
            scope=None,
            confidence=None,
            memory_type=None,
            status=None,
            validity=None,
            supersedes=(),
            evidence=(),
            retrieval_count=None,
            injection_count=None,
            reference_count=None,
            adoption_count=None,
            success_count=None,
            failure_count=None,
            correction_count=None,
            utility=None,
            last_outcome=None,
            layer=layer,
            emit_candidate_trace=True,
        )

    def _has_reusable_scope(self, block: str) -> bool:
        """拒绝只描述当前回合状态、无法跨任务复用的候选。"""
        fields = parse_fields(block)
        scoped_text = " ".join(
            [
                extract_title(block),
                fields.get("context/query", ""),
                fields.get("solution", ""),
                fields.get("takeaways", ""),
            ]
        ).lower()
        ephemeral_markers = (
            "latest user message",
            "latest assistant reply",
            "current turn",
            "this turn",
            "this session only",
            "temporary",
            "temp file",
        )
        return not any(marker in scoped_text for marker in ephemeral_markers)

    def _archive_block(self, block: str, layer: MemoryLayer) -> None:
        """将无效或淘汰块归档到对应层级。"""
        archive_dir = self._archive_dir(layer)
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        archive_file = archive_dir / f"corrupt_{timestamp}.md"
        archive_file.write_text(block, encoding="utf-8")

    # ── 保留与遗忘策略 ──

    def _touch_lru(self, records: Sequence[MemoryRecord]) -> None:
        lru = self._read_lru()
        now = time.time()
        for record in records:
            lru[self._lru_key(record.layer, record.memory_id)] = now
        self._write_lru(lru)

    def _retention_sort_key(
        self,
        record: MemoryRecord,
        *,
        lru: dict[str, float],
    ) -> tuple[float, float]:
        """返回越小越应优先淘汰的 retention key。"""
        status_rank = {
            "superseded": 0.0,
            "obsolete": 0.0,
            "deprecated": 0.0,
            "candidate": 0.5,
            "needs_review": 1.0,
            "active": 2.0,
        }.get(record.status, 1.5)
        validity_rank = {
            "needs_review": 0.0,
            "corrected": 0.0,
            "unknown": 1.0,
            "derived": 1.0,
            "verified": 2.0,
        }.get(record.validity, 1.0)
        type_rank = {
            "episodic": 0.0,
            "preference": 0.5,
            "procedural": 1.5,
            "semantic": 2.0,
        }.get(record.memory_type, 1.0)
        engagement = (
            record.retrieval_count
            + record.injection_count
            + record.reference_count
            + record.adoption_count * 2
        )
        outcome_score = (
            record.success_count - record.failure_count - record.correction_count
        )
        utility_score = max(-4.0, min(4.0, record.utility))
        strength = (
            status_rank * 3.0
            + validity_rank * 2.0
            + type_rank
            + utility_score
            + outcome_score * 0.75
            + min(engagement, 6) * 0.1
        )
        freshness = lru.get(self._lru_key(record.layer, record.memory_id), 0.0)
        return (strength, freshness)

    def _enforce_lru(self, layer: MemoryLayer) -> None:
        if self.max_blocks <= 0:
            return
        records = self.read_memory_records(layer=layer)
        blocks = [record.block for record in records]
        record_keys = {self._lru_key(layer, record.memory_id) for record in records}

        lru = self._read_lru()
        other_layer_lru = {
            key: timestamp
            for key, timestamp in lru.items()
            if not key.startswith(f"{layer}:")
        }
        layer_lru = {
            key: timestamp for key, timestamp in lru.items() if key in record_keys
        }
        cleaned = other_layer_lru | layer_lru
        if cleaned != lru:
            self._write_lru(cleaned)
            lru = cleaned

        if len(blocks) <= self.max_blocks:
            return
        now = time.time()
        for record in records:
            key = self._lru_key(layer, record.memory_id)
            if key not in lru:
                lru[key] = now

        ranked_records = sorted(
            records,
            key=lambda record: self._retention_sort_key(record, lru=lru),
        )
        records_to_evict = {
            record.memory_id
            for record in ranked_records[: len(blocks) - self.max_blocks]
        }

        kept_blocks: list[str] = []
        for record in records:
            key = self._lru_key(layer, record.memory_id)
            if record.memory_id in records_to_evict:
                self._emit_trace(
                    MemoryTraceEvent(
                        type="forgotten",
                        memory_id=record.memory_id,
                        layer=layer,
                        title=record.title,
                        source="retention",
                    )
                )
                self._archive_block(record.block, layer)
            else:
                kept_blocks.append(record.block)

        if len(kept_blocks) < len(blocks):
            self._write_blocks(kept_blocks, layer)
            for record in ranked_records[: len(blocks) - self.max_blocks]:
                lru.pop(self._lru_key(layer, record.memory_id), None)
            self._write_lru(lru)

    def _read_lru(self) -> dict[str, float]:
        if not self.lru_file.exists():
            return {}
        try:
            data = json.loads(self.lru_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    def _write_lru(self, lru: dict[str, float]) -> None:
        self.lru_file.parent.mkdir(parents=True, exist_ok=True)
        self.lru_file.write_text(
            json.dumps(lru, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_blocks(self, blocks: list[str], layer: MemoryLayer) -> None:
        """覆盖写入指定层级的全部记忆块。"""
        content = "\n".join(b.rstrip() for b in blocks if b.strip())
        self._memory_file(layer).write_text(content + "\n", encoding="utf-8")

    def _memory_file(self, layer: MemoryLayer) -> Path:
        """返回指定层级的 MEMORY.md 路径。"""
        if layer == "project":
            return self.memory_file
        return self.user_memory_file

    def _archive_dir(self, layer: MemoryLayer) -> Path:
        """返回指定层级的归档目录。"""
        if layer == "project":
            return self.archive_dir
        return self.user_memory_file.parent / "archive"

    def _selected_layers(
        self,
        layer: MemoryLayerFilter,
    ) -> tuple[MemoryLayer, ...]:
        """将读取过滤器转换为确定的层级顺序。"""
        if layer == "all":
            return ("project", "user")
        return (layer,)

    def _lru_key(self, layer: str, title: str) -> str:
        """生成跨层级无冲突的 LRU 键。"""
        return f"{layer}:{title}"

    def record_injected_records(self, records: Sequence[MemoryRecord]) -> None:
        """记录进入 prompt 上下文的记忆摘要。"""
        self._mark_session_usage(records, usage="injected")
        for record in records:
            self._emit_trace(
                MemoryTraceEvent(
                    type="injected",
                    memory_id=self._memory_id(record.layer, record.title),
                    layer=record.layer,
                    title=record.title,
                    score=record.score,
                    token_count=self._estimate_block_tokens(record.block),
                    source="prompt",
                )
            )

    def record_adopted_records(
        self,
        records: Sequence[MemoryRecord],
        *,
        source: str = "session",
    ) -> None:
        """记录被 agent 明确采用的记忆。"""
        adopted_records: list[MemoryRecord] = []
        for record in records:
            key = (record.layer, record.memory_id)
            state = self._session_usage.setdefault(key, _SessionMemoryUsage())
            if state.adopted:
                continue
            state.adopted = True
            if record.score > 0:
                state.score = record.score
            adopted_records.append(record)
        for record in adopted_records:
            state = self._session_usage[(record.layer, record.memory_id)]
            self._emit_trace(
                MemoryTraceEvent(
                    type="used",
                    memory_id=record.memory_id,
                    layer=record.layer,
                    title=record.title,
                    score=state.score if state.score is not None else record.score,
                    source=source,
                )
            )

    def adopt_injected_records(self, *, source: str = "session") -> int:
        """兼容入口：仅将已确认引用的注入记忆标记为采用。"""
        records_by_layer = {
            current_layer: {
                record.memory_id: record
                for record in self.read_memory_records(layer=current_layer)
            }
            for current_layer in self._selected_layers("all")
        }
        adopted_records: list[MemoryRecord] = []
        for (layer, memory_id), usage in self._session_usage.items():
            if not usage.injected or not usage.referenced or usage.adopted:
                continue
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is None:
                continue
            adopted_records.append(record)
        if not adopted_records:
            return 0
        self.record_adopted_records(adopted_records, source=source)
        return len(adopted_records)

    def record_compaction_referenced_feedback(self) -> int:
        """在 compaction 时把已引用记忆标为采用，结果反馈留到任务结束。"""
        records_by_layer = {
            current_layer: {
                record.memory_id: record
                for record in self.read_memory_records(layer=current_layer)
            }
            for current_layer in self._selected_layers("all")
        }
        updated = 0
        for (layer, memory_id), usage in list(self._session_usage.items()):
            if not usage.referenced or usage.adopted:
                continue
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is None:
                continue
            usage.adopted = True
            self._emit_trace(
                MemoryTraceEvent(
                    type="used",
                    memory_id=record.memory_id,
                    layer=record.layer,
                    title=record.title,
                    score=usage.score,
                    source="compaction",
                )
            )
            updated += 1
        return updated

    def record_llm_references(self, text: str) -> int:
        """使用 LLM reference_judge_fn 检测文本是否引用了已注入的记忆。

        作为 record_explicit_references 的补充，可以捕获隐式引用。
        需要先配置 reference_judge_fn。
        """
        if self.reference_judge_fn is None:
            return 0

        # 收集已注入但未确认引用的记忆
        candidate_keys_by_title: dict[str, list[tuple[str, str]]] = {}
        records_by_layer = {
            current_layer: {
                record.memory_id: record
                for record in self.read_memory_records(layer=current_layer)
            }
            for current_layer in self._selected_layers("all")
        }
        for (layer, memory_id), usage in self._session_usage.items():
            if not usage.injected or usage.referenced:
                continue
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is None:
                continue
            candidate_keys_by_title.setdefault(record.title.casefold(), []).append(
                (layer, memory_id)
            )

        candidate_titles = [
            records_by_layer[layer][memory_id].title
            for keys in candidate_keys_by_title.values()
            if len(keys) == 1
            for layer, memory_id in keys
        ]

        if not candidate_titles:
            return 0

        try:
            matched_titles = self.reference_judge_fn(text, candidate_titles)
        except Exception:
            return 0

        matched = 0
        for title in matched_titles:
            keys = candidate_keys_by_title.get(title.casefold(), [])
            if len(keys) == 1:
                key = keys[0]
                usage = self._session_usage.get(key)
                if usage is not None and not usage.referenced:
                    usage.referenced = True
                    matched += 1
        return matched

    def record_explicit_references(self, text: str) -> int:
        """根据最终回答中的显式 memory_id 或标题标记被引用的记忆。"""
        normalized = text.casefold()
        matched = 0
        records_by_layer = {
            current_layer: {
                record.memory_id: record
                for record in self.read_memory_records(layer=current_layer)
            }
            for current_layer in self._selected_layers("all")
        }
        title_counts: dict[str, int] = {}
        for layer, memory_id in self._session_usage:
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is not None:
                title = record.title.casefold()
                title_counts[title] = title_counts.get(title, 0) + 1
        for (layer, memory_id), usage in self._session_usage.items():
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is None:
                continue
            if memory_id.casefold() in normalized or (
                title_counts.get(record.title.casefold()) == 1
                and record.title.casefold() in normalized
            ):
                if not usage.referenced:
                    matched += 1
                usage.referenced = True
        return matched

    def record_session_outcome(
        self,
        outcome: MemoryOutcome,
        *,
        source: str = "session",
    ) -> int:
        """将本轮 session 中的 memory 使用反馈回写到正式记录。"""
        updated = 0
        records_by_layer = {
            current_layer: {
                record.memory_id: record
                for record in self.read_memory_records(layer=current_layer)
            }
            for current_layer in self._selected_layers("all")
        }
        for (layer, memory_id), usage in list(self._session_usage.items()):
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is None:
                continue
            next_fields = self._feedback_fields_for_record(record, usage, outcome)
            self._replace_record_by_memory_id(record, next_fields)
            updated += 1
        self._session_usage.clear()
        return updated

    def drain_trace_events(self) -> tuple[MemoryTraceEvent, ...]:
        """返回并清空当前进程内累积的 memory trace 事件。"""
        events = tuple(self._trace_events)
        self._trace_events.clear()
        return events

    def _emit_trace(self, event: MemoryTraceEvent) -> None:
        self._trace_events.append(event)

    def _mark_session_usage(
        self,
        records: Sequence[MemoryRecord],
        *,
        usage: Literal["retrieved", "injected", "referenced", "adopted"],
    ) -> None:
        for record in records:
            key = (record.layer, record.memory_id)
            state = self._session_usage.setdefault(key, _SessionMemoryUsage())
            if usage == "retrieved":
                state.retrieved = True
            elif usage == "injected":
                state.injected = True
            elif usage == "referenced":
                state.referenced = True
            elif usage == "adopted":
                state.adopted = True
            if record.score > 0:
                state.score = record.score

    def _feedback_fields_for_record(
        self,
        record: MemoryRecord,
        usage: _SessionMemoryUsage,
        outcome: MemoryOutcome,
    ) -> dict[str, str]:
        retrieval_count = record.retrieval_count + int(usage.retrieved)
        injection_count = record.injection_count + int(usage.injected)
        reference_count = record.reference_count + int(usage.referenced)
        adoption_count = record.adoption_count + int(usage.adopted)
        success_count = record.success_count
        failure_count = record.failure_count
        correction_count = record.correction_count
        utility = record.utility
        status = record.status
        validity = record.validity
        attributed = usage.referenced or usage.adopted
        if attributed:
            if outcome == "success":
                success_count += 1
                utility += 1.0
                if status == "needs_review" and success_count >= failure_count:
                    status = "active"
                if validity in {"needs_review", "corrected"}:
                    validity = "verified"
            elif outcome == "failure":
                failure_count += 1
                utility -= 1.0
                status = "needs_review"
                validity = "needs_review"
            elif outcome == "corrected":
                correction_count += 1
                utility -= 0.5
                status = "needs_review"
                validity = "corrected"
        return {
            "retrieval-count": str(retrieval_count),
            "injection-count": str(injection_count),
            "reference-count": str(reference_count),
            "adoption-count": str(adoption_count),
            "success-count": str(success_count),
            "failure-count": str(failure_count),
            "correction-count": str(correction_count),
            "utility": f"{utility:.2f}",
            "last-outcome": outcome if attributed else (record.last_outcome or ""),
            "status": status,
            "validity": validity,
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }

    def _replace_record_by_memory_id(
        self,
        record: MemoryRecord,
        field_updates: dict[str, str],
    ) -> None:
        blocks = self.read_memory_blocks(layer=cast("MemoryLayerFilter", record.layer))
        updated: list[str] = []
        replaced = False
        for block in blocks:
            existing_record = parse_memory_record(
                block, layer=cast("MemoryLayer", record.layer)
            )
            if existing_record.memory_id == record.memory_id:
                updated.append(
                    self._rewrite_record_block(existing_record, field_updates)
                )
                replaced = True
            else:
                updated.append(existing_record.block.rstrip() + "\n")
        if replaced:
            self._write_blocks(updated, cast("MemoryLayer", record.layer))

    def _rewrite_record_block(
        self,
        record: MemoryRecord,
        field_updates: dict[str, str],
    ) -> str:
        fields = dict(record.fields)
        fields.update(field_updates)
        lines = [f"## {record.title}"]
        mandatory = ("context/query", "solution", "files", "takeaways")
        display_names = {
            "context/query": "Context/Query",
            "solution": "Solution",
            "files": "Files",
            "takeaways": "Takeaways",
        }
        for key in mandatory:
            value = fields.pop(key, "").strip()
            if value:
                lines.append(f"- {display_names[key]}: {value}")
        for key, value in fields.items():
            value = value.strip()
            if not value:
                continue
            lines.append(f"- {self._display_field_name(key)}: {value}")
        return "\n".join(lines).strip() + "\n"

    def _display_field_name(self, key: str) -> str:
        parts = [part.capitalize() for part in key.split("-")]
        return "-".join(parts)

    def _estimate_block_tokens(self, block: str) -> int:
        from xcode.agent._compaction import estimate_tokens

        return estimate_tokens(block)

    def _memory_id(self, layer: str, title: str) -> str:
        return build_memory_id(layer=layer, title=title)

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 3)

    # ── 混合检索（语义 + BM25）──

    def hybrid_search_memory_records(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
        layer: MemoryLayerFilter = "all",
        *,
        source: str = "api",
        track_usage: bool = True,
        retrieval_context: MemoryRetrievalContext | None = None,
        semantic_weight: float = _DEFAULT_SEMANTIC_WEIGHT,
    ) -> list[MemoryRecord]:
        """兼容旧 API；委托给统一检索流水线。"""
        return self._search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=layer,
            source=source,
            track_usage=track_usage,
            retrieval_context=retrieval_context,
            semantic_weight=semantic_weight,
        )

    def render_prompt_packet(self, record: MemoryRecord) -> str:
        """将记忆渲染为短小、可审计的 prompt packet。"""
        lines = [
            (
                f'<record id="{escape(record.memory_id)}" '
                f'type="{escape(record.memory_type)}" '
                f'layer="{escape(record.layer)}" '
                f'score="{record.score:.3f}">'
            ),
            f"<conclusion>{escape(self._conclusion_text(record))}</conclusion>",
        ]
        if record.scope:
            lines.append(f"<scope>{escape(record.scope)}</scope>")
        evidence_summary = self._evidence_summary(record)
        if evidence_summary:
            lines.append(f"<evidence>{escape(evidence_summary)}</evidence>")
        source_summary = self._source_summary(record)
        if source_summary:
            lines.append(f"<source>{escape(source_summary)}</source>")
        lines.append("</record>")
        return "\n".join(lines)

    def render_search_result(self, record: MemoryRecord) -> str:
        """渲染显式检索结果，保留完整记录并补充结构化头部。"""
        lines = [
            (
                f"[{record.layer}] id={record.memory_id} type={record.memory_type} "
                f"score={record.score:.3f}"
            ),
            f"title: {record.title}",
        ]
        evidence_summary = self._evidence_summary(record)
        if evidence_summary:
            lines.append(f"evidence: {evidence_summary}")
        lines.append(record.block.strip())
        return "\n".join(lines)

    def _conclusion_text(self, record: MemoryRecord) -> str:
        for key in ("solution", "takeaways"):
            value = record.fields.get(key, "").strip()
            if value:
                return value
        return record.title

    def _evidence_summary(self, record: MemoryRecord) -> str:
        if record.evidence:
            return "; ".join(
                f"{item.kind}:{item.reference}" for item in record.evidence[:3]
            )
        fallback = []
        for key in ("validated", "validation", "source-session", "source"):
            value = record.fields.get(key, "").strip()
            if value:
                fallback.append(value)
        return "; ".join(fallback[:2])

    def _source_summary(self, record: MemoryRecord) -> str:
        if record.source_session:
            return f"{record.layer}:{record.source_session}"
        return record.layer

    def get_last_used_at(self, record: MemoryRecord) -> float | None:
        """返回某条记忆最近一次被检索使用的时间戳。"""
        return self._read_lru().get(self._lru_key(record.layer, record.memory_id))

    def _passes_confidence_gate(self, record: MemoryRecord) -> bool:
        if record.confidence_value is None:
            return True
        return record.confidence_value >= self.min_confidence


# ── 模块级辅助函数 ──


_NGRAM_DIM = 256


def _ngram_hash_vector(text: str, dim: int = _NGRAM_DIM) -> list[float]:
    """字符 n-gram 哈希向量，零依赖的语义嵌入。"""
    normalized = text.lower().strip()
    if not normalized:
        return [0.0] * dim
    vec = [0.0] * dim
    for i in range(len(normalized) - 1):
        ngram = normalized[i : i + 2]
        idx = hash(ngram) % dim
        vec[idx] += 1.0
    for i in range(len(normalized) - 2):
        ngram = normalized[i : i + 3]
        idx = hash(ngram) % dim
        vec[idx] += 1.0
    for token in normalized.split():
        idx = hash(token) % dim
        vec[idx] += 2.0
    vec = [max(0.0, v) ** 0.5 for v in vec]
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 1e-10:
        vec = [v / norm for v in vec]
    return vec


def build_memory_embedding_fn() -> MemoryEmbeddingFn:
    """构建语义嵌入函数。

    优先使用 sentence-transformers（需独立安装），
    回退到零依赖的字符 n-gram 哈希向量。
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]  # not available in CI

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(text: str) -> list[float]:
            return model.encode(text).tolist()

        return embed
    except ImportError:
        import logging

        logging.getLogger(__name__).info(
            "sentence-transformers not installed; using n-gram hash embedding. "
            "Install with: pip install sentence-transformers"
        )
        return _ngram_hash_vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def build_memory_judge_fn(
    llm: object,
    *,
    model: str | None = None,
    provider_id: str = "main",
) -> MemoryJudgeFn:
    """使用 LLM provider 构建记忆质量评估函数。

    生成一个 MemoryJudgeFn，对传入的决策文本调用 provider 做结构化判断。
    使用 JSON mode 输出，确保 LLM 返回可解析的结构化结果。

    参数：
        llm: 实现了 ModelProvider 协议的 LLM provider 实例。
        model: 可选，指定使用的模型名（如 "deepseek-chat"）。
              为 None 时使用 provider 的默认模型。
        provider_id: 可选的 provider 标识，仅用于 trace 事件。

    返回：
        一个 MemoryJudgeFn 可调用对象。
    """
    system_prompt = (
        "You are a memory quality judge. Your task is to evaluate whether a single "
        "decision extracted from a conversation is worth remembering for future sessions.\n\n"
        "Output JSON with exactly these fields:\n"
        '{"is_worth_remembering": true/false, '
        '"confidence": 0.0-1.0, '
        '"scope": "subsystem name or null", '
        '"related_files": ["file1.py"], '
        '"suggested_title": "Decision: <short title>", '
        '"suggested_context": "What problem or question led to this decision", '
        '"suggested_solution": "What was decided", '
        '"suggested_takeaways": "Why this matters and what to remember", '
        '"reasoning": "Brief explanation of the judgment"}\n\n'
        "Guidelines:\n"
        "- Set is_worth_remembering=false if: ephemeral/temp state, obvious fixes, "
        "personal preference, trivial config changes.\n"
        "- Set is_worth_remembering=true if: architecture decisions, API design choices, "
        "non-obvious bug fixes, dependency choices, important constraints.\n"
        "- confidence should reflect how certain you are about the judgment.\n"
        "- scope should be the subsystem or module this decision affects, or null.\n"
        "- related_files should list specific files involved, if inferrable.\n"
        "- suggested_title should start with 'Decision: ' and be concise.\n"
        "- Output ONLY valid JSON, no other text.\n"
    )

    def judge(decision_text: str) -> MemoryJudgeResult:
        """调用 provider 评判单条决策（同步接口，兼容运行中的 event loop）。"""
        if not decision_text.strip() or len(decision_text.strip()) < 10:
            return MemoryJudgeResult(
                is_worth_remembering=False,
                confidence=0.0,
                reasoning="decision text too short",
            )

        try:
            import asyncio

            user_prompt = f"Evaluate this decision:\n{decision_text}"

            # 在已有事件循环中使用 run_coroutine_threadsafe，
            # 否则使用 asyncio.run（同步上下文）
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    _judge_async(llm, system_prompt, user_prompt),
                    loop,
                )
                raw = future.result(timeout=30)
            except RuntimeError:
                raw = asyncio.run(_judge_async(llm, system_prompt, user_prompt))

            result = _parse_judge_json(raw)
            return result
        except Exception as exc:
            # LLM 调用失败时保守降级：记录但不拒绝
            return MemoryJudgeResult(
                is_worth_remembering=True,
                confidence=0.3,
                reasoning=f"LLM judge failed, conservative fallback: {exc}",
            )

    return judge


def build_memory_consolidate_judge_fn(
    llm: object,
) -> MemoryConsolidateJudgeFn:
    """构建节级 LLM 记忆质量评估函数。

    与 build_memory_judge_fn 不同，此函数接收完整的 Key Decisions 节文本，
    LLM 一次看到所有决策的前后文，可以合并相关条目、过滤低价值决策、重组表述。

    输出 JSON 数组，每个元素与 MemoryJudgeResult 的字段一致。
    """
    system_prompt = (
        "You are a memory consolidation judge. Review the full list of decisions "
        "extracted from a conversation session and decide which are worth remembering.\n\n"
        "Rules:\n"
        "- Merge related decisions into one record when they are about the same topic.\n"
        "- Drop ephemeral/obvious/trivial decisions (temp fixes, obvious config changes).\n"
        "- Keep architecture decisions, API design choices, non-obvious bug fixes, "
        "dependency choices, important constraints.\n"
        "- For each kept decision, rewrite it clearly for future retrieval.\n\n"
        "Output a JSON array. Each element has these fields:\n"
        '{"is_worth_remembering": true/false, '
        '"confidence": 0.0-1.0, '
        '"scope": "subsystem name or null", '
        '"related_files": ["file1.py"], '
        '"suggested_title": "Decision: <concise title>", '
        '"suggested_context": "What problem or question led to this decision", '
        '"suggested_solution": "What was decided", '
        '"suggested_takeaways": "Why this matters", '
        '"reasoning": "Why kept/merged/dropped"}\n\n'
        "Output ONLY valid JSON array, no other text.\n"
    )

    def judge_section(section_text: str) -> list[MemoryJudgeResult]:
        if not section_text.strip():
            return []
        try:
            import asyncio

            user_prompt = f"Consolidate these decisions:\n\n{section_text}"
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    _judge_async(llm, system_prompt, user_prompt),
                    loop,
                )
                raw = future.result(timeout=60)
            except RuntimeError:
                raw = asyncio.run(_judge_async(llm, system_prompt, user_prompt))

            return _parse_judge_json_array(raw)
        except Exception:
            return []

    return judge_section


def build_memory_reference_judge_fn(
    llm: object,
) -> MemoryReferenceJudgeFn:
    """构建 LLM 引用检测函数。

    接收(compaction 摘要文本, 候选记忆标题列表)，
    返回被引用的记忆标题列表。用于补充纯子串匹配的不足。
    """
    system_prompt = (
        "You are a reference detector. Given a text summary and a list of known "
        "memory records (each with a title), determine which memory records the "
        "summary implicitly or explicitly references.\n\n"
        "Output ONLY a JSON array of referenced memory titles, e.g. "
        '["Decision: Use Redis", "Decision: Layered architecture"].\n'
        "Return empty array [] if none are referenced.\n"
        "Be inclusive: if the summary uses the same concept, decision, or finding "
        "as a memory record, consider it referenced even if the exact title is not used.\n"
    )

    def detect_references(text: str, candidates: list[str]) -> list[str]:
        if not text.strip() or not candidates:
            return []
        try:
            import asyncio

            candidates_text = "\n".join(f"- {c}" for c in candidates)
            user_prompt = (
                f"Text:\n{text[:2000]}\n\n"
                f"Memory records:\n{candidates_text}\n\n"
                f"Which memory titles are referenced in the text?"
            )
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    _judge_async(llm, system_prompt, user_prompt),
                    loop,
                )
                raw = future.result(timeout=30)
            except RuntimeError:
                raw = asyncio.run(_judge_async(llm, system_prompt, user_prompt))

            return _parse_reference_json_list(raw, candidates)
        except Exception:
            return []

    return detect_references


async def _judge_async(
    llm: object,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """异步调用 provider 进行评判。"""
    from xcode.ai.events import TextDelta
    from xcode.ai.types import StreamOptions

    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text_parts: list[str] = []
        async for event in llm.stream(  # type: ignore[union-attr]
            messages=messages,
            tools=[],
            options=StreamOptions(max_tokens=1024),
        ):
            if isinstance(event, TextDelta):
                text_parts.append(event.chunk)
        return "".join(text_parts)
    except Exception:
        return json.dumps(
            {
                "is_worth_remembering": True,
                "confidence": 0.3,
                "scope": None,
                "related_files": [],
                "suggested_title": "",
                "suggested_context": "",
                "suggested_solution": "",
                "suggested_takeaways": "",
                "reasoning": "sync fallback: provider stream failed",
            }
        )


def _parse_judge_json(raw: str) -> MemoryJudgeResult:
    """解析 LLM 返回的 JSON，提取 MemoryJudgeResult。"""
    import json
    import re

    # 尝试解析 JSON（处理可能的 markdown 包裹）
    text = raw.strip()
    # 移除 ```json 和 ``` 包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON 块
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return MemoryJudgeResult(
        is_worth_remembering=bool(data.get("is_worth_remembering", True)),
        confidence=float(data.get("confidence", 0.5)),
        scope=str(data.get("scope")) if data.get("scope") else None,
        related_files=tuple(str(f) for f in (data.get("related_files") or []) if f),
        suggested_title=str(data.get("suggested_title", "")),
        suggested_context=str(data.get("suggested_context", "")),
        suggested_solution=str(data.get("suggested_solution", "")),
        suggested_takeaways=str(data.get("suggested_takeaways", "")),
        reasoning=str(data.get("reasoning", "")),
    )


def _parse_judge_json_array(raw: str) -> list[MemoryJudgeResult]:
    """解析 LLM 返回的 JSON 数组，提取 MemoryJudgeResult 列表。"""
    import json
    import re

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data_list = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text, re.DOTALL)
        if match:
            try:
                data_list = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                data_list = []
        else:
            data_list = []

    if not isinstance(data_list, list):
        return []

    results: list[MemoryJudgeResult] = []
    for item in data_list:
        if not isinstance(item, dict):
            continue
        results.append(
            MemoryJudgeResult(
                is_worth_remembering=bool(item.get("is_worth_remembering", True)),
                confidence=float(item.get("confidence", 0.5)),
                scope=str(item.get("scope")) if item.get("scope") else None,
                related_files=tuple(
                    str(f) for f in (item.get("related_files") or []) if f
                ),
                suggested_title=str(item.get("suggested_title", "")),
                suggested_context=str(item.get("suggested_context", "")),
                suggested_solution=str(item.get("suggested_solution", "")),
                suggested_takeaways=str(item.get("suggested_takeaways", "")),
                reasoning=str(item.get("reasoning", "")),
            )
        )
    return results


def _parse_reference_json_list(raw: str, candidates: list[str]) -> list[str]:
    """解析 LLM 返回的 JSON 引用标题列表。"""
    import json
    import re

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        titles = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text, re.DOTALL)
        if match:
            try:
                titles = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                titles = []
        else:
            titles = []

    if not isinstance(titles, list):
        return []

    # 只返回与候选匹配的标题（模糊匹配）
    result: list[str] = []
    candidate_lower = {c.lower(): c for c in candidates}
    for title in titles:
        t = str(title).strip()
        if t.lower() in candidate_lower:
            result.append(candidate_lower[t.lower()])
        else:
            # 尝试部分匹配
            for c_lower, c_orig in candidate_lower.items():
                if t.lower() in c_lower or c_lower in t.lower():
                    result.append(c_orig)
                    break
    return result
