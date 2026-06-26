"""基于 BM25 的 MEMORY.md 记忆系统。

支持质量门（拒绝低质量/重复块）、冲突合并（同标题合并字段）、
保留策略（超过 max_blocks 时结合质量与使用信号淘汰较弱记忆）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

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


@dataclass
class _SessionMemoryUsage:
    retrieved: bool = False
    injected: bool = False
    referenced: bool = False
    adopted: bool = False


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
    ) -> None:
        """创建项目级与用户级并行的记忆管理器。"""
        self.root = root
        self.memory_file = root / "MEMORY.md"
        self.user_memory_file = user_memory_file or (
            Path.home() / ".xcode" / "memory" / "MEMORY.md"
        )
        self.archive_dir = root / ".local" / "memory_archive"
        self.candidate_dir = root / ".local" / "memory_candidates"
        self.quarantine_dir = root / ".local" / "memory_quarantine"
        self.lru_file = root / ".local" / "memory_lru.json"
        self.max_blocks = max_blocks
        self.min_retrieval_score = min_retrieval_score
        self.min_confidence = min_confidence
        self.rerank_policy = rerank_policy or MemoryRerankPolicy()
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
    ) -> list[str]:
        """跨项目级与用户级记忆检索匹配块。"""
        records = self.search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=layer,
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
    ) -> list[MemoryRecord]:
        """跨层级执行检索和重排并返回带来源的记录。"""
        started_at = time.perf_counter()
        candidates = self.retrieve_memory_candidates(query, layer=layer)
        if not candidates or limit <= 0:
            if source == "tool":
                self._emit_trace(
                    MemoryTraceEvent(
                        type="tool_searched",
                        latency_ms=self._elapsed_ms(started_at),
                        source=source,
                    )
                )
            return []

        ranked = self.rerank_memory_candidates(
            candidates,
            query,
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
            if track_usage:
                self._emit_trace(
                    MemoryTraceEvent(
                        type="used",
                        memory_id=record.memory_id,
                        layer=record.layer,
                        title=record.title,
                        score=record.score,
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
        query: str,
        *,
        layer: MemoryLayerFilter = "all",
    ) -> list[MemoryRecord]:
        """返回 lexical candidate 集，供后续 rerank 使用。"""
        records = self.read_memory_records(layer=layer)
        blocks = [record.block for record in records]
        if not blocks or not query.strip():
            return []

        corpus = [tokenize(block) for block in blocks]
        query_words = tokenize(query)
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
            lexical = self._weighted_lexical_score(record, query, bm25_score=score)
            candidates.append(
                MemoryRecord(
                    block=record.block,
                    title=record.title,
                    fields=record.fields,
                    memory_id=record.memory_id,
                    memory_type=record.memory_type,
                    scope=record.scope,
                    source_session=record.source_session,
                    related_files=record.related_files,
                    related_symbols=record.related_symbols,
                    created_at=record.created_at,
                    modified_at=record.modified_at,
                    confidence_value=record.confidence_value,
                    status=record.status,
                    validity=record.validity,
                    supersedes=record.supersedes,
                    evidence=record.evidence,
                    retrieval_count=record.retrieval_count,
                    injection_count=record.injection_count,
                    reference_count=record.reference_count,
                    adoption_count=record.adoption_count,
                    success_count=record.success_count,
                    failure_count=record.failure_count,
                    correction_count=record.correction_count,
                    utility=record.utility,
                    last_outcome=record.last_outcome,
                    score=round(lexical, 6),
                    layer=record.layer,
                )
            )
        candidates.sort(key=lambda r: (-r.score, r.title))
        return candidates

    def rerank_memory_candidates(
        self,
        candidates: Sequence[MemoryRecord],
        query: str,
        *,
        scope: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """对 lexical candidates 应用可替换的 rerank 与 gate。"""
        ranked: list[MemoryRecord] = []
        for candidate in candidates:
            adjusted = self._apply_rerank_policy(candidate, query, scope)
            if adjusted < self.min_retrieval_score or not self._passes_confidence_gate(
                candidate
            ):
                continue
            ranked.append(
                MemoryRecord(
                    block=candidate.block,
                    title=candidate.title,
                    fields=candidate.fields,
                    memory_id=candidate.memory_id,
                    memory_type=candidate.memory_type,
                    scope=candidate.scope,
                    source_session=candidate.source_session,
                    related_files=candidate.related_files,
                    related_symbols=candidate.related_symbols,
                    created_at=candidate.created_at,
                    modified_at=candidate.modified_at,
                    confidence_value=candidate.confidence_value,
                    status=candidate.status,
                    validity=candidate.validity,
                    supersedes=candidate.supersedes,
                    evidence=candidate.evidence,
                    retrieval_count=candidate.retrieval_count,
                    injection_count=candidate.injection_count,
                    reference_count=candidate.reference_count,
                    adoption_count=candidate.adoption_count,
                    success_count=candidate.success_count,
                    failure_count=candidate.failure_count,
                    correction_count=candidate.correction_count,
                    utility=candidate.utility,
                    last_outcome=candidate.last_outcome,
                    score=round(adjusted, 6),
                    layer=candidate.layer,
                )
            )
        ranked.sort(key=lambda r: (-r.score, r.title))
        if limit is not None and limit > 0:
            return ranked[:limit]
        return ranked

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
            results.append(
                MemorySearchEvalResult(
                    query=case.query,
                    passed=any(expected in title for title in titles),
                    expected_title_contains=expected,
                    matched_titles=titles,
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
        query: str,
        *,
        bm25_score: float,
    ) -> float:
        query_terms = tokenize_set(query)
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
        score += self._exact_match_bonus(record, query)
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
        if record.failure_count > record.success_count or record.last_outcome == "failure":
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
        if len(tokenize(query)) >= 2 and any(normalized_query in field for field in phrase_fields):
            bonus += self.rerank_policy.phrase_match_bonus
        return bonus

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
            self._emit_trace(
                MemoryTraceEvent(
                    type="candidate_created",
                    memory_id=self._memory_id(layer, title) if title else None,
                    layer=layer,
                    title=title or None,
                    source=source,
                )
            )
        if not self.validate_memory_block(block):
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
            merged_block = self._merge_with_existing(block, new_title, existing_records)
            if merged_block is not None:
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

        rejection_reason = self._quality_rejection_reason(block, existing_records)
        if rejection_reason is not None:
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
            return False

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

    def migrate_legacy_records(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> int:
        """一次性补齐缺失的 memory_id / type / status / validity 元数据。"""
        updated_count = 0
        for current_layer in self._selected_layers(layer):
            blocks = self.read_memory_blocks(layer=current_layer)
            rewritten: list[str] = []
            changed = False
            for block in blocks:
                normalized = (
                    with_metadata(
                        block,
                        layer=current_layer,
                        source=None,
                        scope=None,
                        confidence=None,
                        status="active",
                        validity="unknown",
                    ).strip()
                    + "\n"
                )
                rewritten.append(normalized)
                if normalized != block:
                    changed = True
                    updated_count += 1
            if changed:
                self._write_blocks(rewritten, current_layer)
        return updated_count

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

    def consolidate(self, summary: str) -> None:
        """从压缩摘要中提取候选记忆，并经 gate 决定晋升或隔离。"""
        for block in self._extract_summary_blocks(summary):
            if self._is_memory_attempt(block):
                self._ingest_consolidation_candidate(
                    block,
                    source="consolidation",
                    layer="project",
                )

    def _extract_summary_blocks(self, summary: str) -> list[str]:
        """从 compact summary 中提取 H2 结构块。"""
        content = summary.removeprefix("[Compressed]").strip()
        if "## " not in content:
            return []
        normalized_lines: list[str] = []
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("- ") and "## " in stripped:
                stripped = stripped[stripped.index("## ") :]
            normalized_lines.append(stripped)
        normalized = "\n".join(normalized_lines)
        normalized = re.sub(
            r"\s-\s(?=[A-Za-z][A-Za-z0-9_/-]*:)",
            "\n- ",
            normalized,
        )
        blocks: list[str] = []
        for part in normalized.split("## "):
            fragment = part.strip()
            if not fragment:
                continue
            block = "## " + fragment
            blocks.append(
                "\n".join(line.strip() for line in block.splitlines() if line.strip())
            )
        return blocks

    def _is_memory_attempt(self, block: str) -> bool:
        has_field = any(
            f in block for f in ["Context/Query", "Solution", "Files", "Takeaways"]
        )
        return block.strip().startswith("## ") and (
            has_field or "incident" in block.lower()
        )

    def _ingest_consolidation_candidate(
        self,
        block: str,
        *,
        source: str,
        layer: MemoryLayer,
    ) -> None:
        """将 compaction 产物写成候选，并执行晋升 gate。"""
        title = extract_title(block)
        memory_id = self._memory_id(layer, title) if title else None
        self._write_candidate_block(block, layer=layer, source=source)
        self._emit_trace(
            MemoryTraceEvent(
                type="candidate_created",
                memory_id=memory_id,
                layer=layer,
                title=title or None,
                source=source,
            )
        )
        existing_records = self.read_memory_records(layer=layer)
        rejection_reason = self._promotion_gate_rejection_reason(
            block,
            existing_records=existing_records,
        )
        if rejection_reason is not None:
            self._quarantine_block(
                block,
                layer=layer,
                reason=rejection_reason,
                source=source,
            )
            self._emit_trace(
                MemoryTraceEvent(
                    type="quarantined",
                    memory_id=memory_id,
                    layer=layer,
                    title=title or None,
                    rejection_reason=rejection_reason,
                    source=source,
                )
            )
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
            emit_candidate_trace=False,
        )

    def _promotion_gate_rejection_reason(
        self,
        block: str,
        *,
        existing_records: list[MemoryRecord],
    ) -> str | None:
        """晋升 gate：在写入正式记忆前执行结构、质量与证据检查。"""
        if not self.validate_memory_block(block):
            return "schema_validation_failed"
        quality_rejection = self._quality_rejection_reason(block, existing_records)
        if quality_rejection is not None:
            return quality_rejection
        if not self._has_reusable_scope(block):
            return "scope_gate_failed"
        if not self._has_promotable_evidence(block):
            return "evidence_gate_failed"
        return None

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

    def _has_promotable_evidence(self, block: str) -> bool:
        """要求候选至少带有可审计的 outcome / evidence / validation 信号。"""
        fields = parse_fields(block)
        evidence_fields = (
            "evidence",
            "validated",
            "validation",
            "result",
            "outcome",
            "confidence",
        )
        for key in evidence_fields:
            value = fields.get(key, "").strip()
            if value:
                return True
        combined = " ".join(fields.values()).lower()
        evidence_markers = (
            "passed",
            "verified",
            "confirmed",
            "reproduced",
            "failed because",
            "test:",
            "pytest",
        )
        return any(marker in combined for marker in evidence_markers)

    def _archive_block(self, block: str, layer: MemoryLayer) -> None:
        """将无效或淘汰块归档到对应层级。"""
        archive_dir = self._archive_dir(layer)
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        archive_file = archive_dir / f"corrupt_{timestamp}.md"
        archive_file.write_text(block, encoding="utf-8")

    def _write_candidate_block(
        self,
        block: str,
        *,
        layer: MemoryLayer,
        source: str,
    ) -> Path:
        """将待晋升候选持久化，保留 compaction 原文以便审查。"""
        candidate_dir = self._candidate_dir(layer)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        slug = self._candidate_slug(extract_title(block)) or f"candidate-{timestamp}"
        candidate_file = candidate_dir / f"{timestamp}_{slug}.md"
        payload = (
            f"- Source: {source}\n"
            f"- Created: {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}\n"
            "\n"
            f"{block.strip()}\n"
        )
        candidate_file.write_text(payload, encoding="utf-8")
        return candidate_file

    def _quarantine_block(
        self,
        block: str,
        *,
        layer: MemoryLayer,
        reason: str,
        source: str,
    ) -> None:
        """保留未通过晋升 gate 的候选，避免污染正式记忆。"""
        quarantine_dir = self._quarantine_dir(layer)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        slug = self._candidate_slug(extract_title(block)) or f"candidate-{timestamp}"
        quarantine_file = quarantine_dir / f"{timestamp}_{slug}.md"
        payload = (
            f"- Source: {source}\n"
            f"- Reason: {reason}\n"
            f"- Created: {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}\n"
            "\n"
            f"{block.strip()}\n"
        )
        quarantine_file.write_text(payload, encoding="utf-8")

    def _candidate_slug(self, title: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        return normalized[:48]

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
        outcome_score = record.success_count - record.failure_count - record.correction_count
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
            record.memory_id for record in ranked_records[: len(blocks) - self.max_blocks]
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

    def _candidate_dir(self, layer: MemoryLayer) -> Path:
        """返回指定层级的候选目录。"""
        if layer == "project":
            return self.candidate_dir
        return self.user_memory_file.parent / "candidates"

    def _quarantine_dir(self, layer: MemoryLayer) -> Path:
        """返回指定层级的隔离目录。"""
        if layer == "project":
            return self.quarantine_dir
        return self.user_memory_file.parent / "quarantine"

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

    def record_adopted_records(self, records: Sequence[MemoryRecord]) -> None:
        """记录被 agent 明确采用的记忆。"""
        self._mark_session_usage(records, usage="adopted")

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
        for (layer, memory_id), usage in self._session_usage.items():
            record = records_by_layer.get(layer, {}).get(memory_id)
            if record is None:
                continue
            if memory_id.casefold() in normalized or record.title.casefold() in normalized:
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
        touched_layers = {
            layer for (layer, _memory_id), _usage in self._session_usage.items()
        }
        self._session_usage.clear()
        for layer in touched_layers:
            self.derive_procedural_candidates(layer=layer)
        return updated

    def derive_procedural_candidates(
        self,
        *,
        layer: MemoryLayer = "project",
    ) -> int:
        """从稳定成功的 episodic 记录中提炼 procedural candidate。"""
        records = self.read_memory_records(layer=layer)
        groups: dict[tuple[str, str, str], list[MemoryRecord]] = defaultdict(list)
        for record in records:
            if record.memory_type != "episodic":
                continue
            groups[self._procedural_group_key(record)].append(record)

        created = 0
        for grouped in groups.values():
            successful = [record for record in grouped if self._is_successful_episode(record)]
            if len(successful) < 2:
                continue
            candidate_block = self._build_procedural_candidate(successful)
            title = extract_title(candidate_block)
            if self._candidate_exists(title, layer=layer):
                continue
            counterexamples = [record for record in grouped if self._has_counterexample(record)]
            self._emit_trace(
                MemoryTraceEvent(
                    type="candidate_created",
                    memory_id=self._memory_id(layer, title) if title else None,
                    layer=layer,
                    title=title or None,
                    source="procedural_promotion",
                )
            )
            if counterexamples:
                quarantined = self._append_counterexamples(
                    candidate_block,
                    counterexamples=counterexamples,
                )
                self._quarantine_block(
                    quarantined,
                    layer=layer,
                    reason="counterexample_gate_failed",
                    source="procedural_promotion",
                )
                self._emit_trace(
                    MemoryTraceEvent(
                        type="quarantined",
                        memory_id=self._memory_id(layer, title) if title else None,
                        layer=layer,
                        title=title or None,
                        rejection_reason="counterexample_gate_failed",
                        source="procedural_promotion",
                    )
                )
                continue
            self._write_candidate_block(
                candidate_block,
                layer=layer,
                source="procedural_promotion",
            )
            created += 1
        return created

    def promote_candidate(
        self,
        title: str,
        *,
        layer: MemoryLayer = "project",
    ) -> bool:
        """将候选目录中的记忆晋升为正式 memory 记录。"""
        candidate_file = self._find_candidate_file(title, layer=layer)
        if candidate_file is None:
            return False
        candidate_block = self._extract_candidate_block(
            candidate_file.read_text(encoding="utf-8")
        )
        existing_records = self.read_memory_records(layer=layer)
        rejection_reason = self._promotion_gate_rejection_reason(
            candidate_block,
            existing_records=existing_records,
        )
        if rejection_reason is not None:
            self._move_candidate_to_quarantine(
                candidate_file,
                layer=layer,
                reason=rejection_reason,
                source="candidate_review",
            )
            self._emit_trace(
                MemoryTraceEvent(
                    type="quarantined",
                    memory_id=self._memory_id(layer, title),
                    layer=layer,
                    title=title,
                    rejection_reason=rejection_reason,
                    source="candidate_review",
                )
            )
            return False
        promoted_block = self._normalize_candidate_for_promotion(candidate_block)
        persisted = self._persist_memory_block(
            promoted_block,
            source="candidate_review",
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
            emit_candidate_trace=False,
        )
        if persisted:
            candidate_file.unlink(missing_ok=True)
        return persisted

    def reject_candidate(
        self,
        title: str,
        *,
        layer: MemoryLayer = "project",
        reason: str = "manual_rejection",
    ) -> bool:
        """将候选目录中的记忆移入 quarantine。"""
        candidate_file = self._find_candidate_file(title, layer=layer)
        if candidate_file is None:
            return False
        self._move_candidate_to_quarantine(
            candidate_file,
            layer=layer,
            reason=reason,
            source="candidate_review",
        )
        self._emit_trace(
            MemoryTraceEvent(
                type="quarantined",
                memory_id=self._memory_id(layer, title),
                layer=layer,
                title=title,
                rejection_reason=reason,
                source="candidate_review",
            )
        )
        return True

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
        if usage.adopted:
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
            "last-outcome": outcome,
            "status": status,
            "validity": validity,
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }

    def _replace_record_by_memory_id(
        self,
        record: MemoryRecord,
        field_updates: dict[str, str],
    ) -> None:
        blocks = self.read_memory_blocks(layer=record.layer)
        updated: list[str] = []
        replaced = False
        for block in blocks:
            existing_record = parse_memory_record(block, layer=record.layer)
            if existing_record.memory_id == record.memory_id:
                updated.append(
                    self._rewrite_record_block(existing_record, field_updates)
                )
                replaced = True
            else:
                updated.append(existing_record.block.rstrip() + "\n")
        if replaced:
            self._write_blocks(updated, record.layer)

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

    def _procedural_group_key(self, record: MemoryRecord) -> tuple[str, str, str]:
        scope = (record.scope or "").strip().lower()
        files = ",".join(item.strip().lower() for item in record.related_files)
        solution = record.fields.get("solution", "").strip().lower()
        return (scope, files, solution)

    def _is_successful_episode(self, record: MemoryRecord) -> bool:
        return (
            record.success_count > 0
            and record.failure_count == 0
            and record.correction_count == 0
            and record.status == "active"
            and record.validity not in {"needs_review", "corrected"}
        )

    def _has_counterexample(self, record: MemoryRecord) -> bool:
        return record.failure_count > 0 or record.correction_count > 0

    def _build_procedural_candidate(self, records: Sequence[MemoryRecord]) -> str:
        anchor = sorted(records, key=lambda record: record.title)[0]
        solution = anchor.fields.get("solution", "").strip()
        takeaways = anchor.fields.get("takeaways", "").strip()
        files = anchor.fields.get("files", "").strip()
        context = "; ".join(record.title for record in records)
        title = f"How to: {solution}".strip()
        source_records = ", ".join(record.memory_id for record in records)
        evidence = "; ".join(f"memory:{record.memory_id}" for record in records)
        lines = [
            f"## {title}",
            f"- Context/Query: Abstracted from successful incidents: {context}",
            f"- Solution: {solution}",
            f"- Files: {files}",
            f"- Takeaways: {takeaways}",
            "- Memory-Type: procedural",
            f"- Scope: {anchor.scope or 'project'}",
            f"- Source-Records: {source_records}",
            f"- Evidence: {evidence}",
            "- Status: candidate",
            "- Validity: derived",
        ]
        return "\n".join(lines) + "\n"

    def _append_counterexamples(
        self,
        block: str,
        *,
        counterexamples: Sequence[MemoryRecord],
    ) -> str:
        ids = ", ".join(record.memory_id for record in counterexamples)
        return block.strip() + f"\n- Counterexamples: {ids}\n"

    def _candidate_exists(
        self,
        title: str,
        *,
        layer: MemoryLayer,
    ) -> bool:
        normalized = title.strip().lower()
        if not normalized:
            return False
        for record in self.read_memory_records(layer=layer):
            if record.title.strip().lower() == normalized:
                return True
        for root_dir in (self._candidate_dir(layer), self._quarantine_dir(layer)):
            for path in root_dir.glob("*.md"):
                if f"## {title}" in path.read_text(encoding="utf-8"):
                    return True
        return False

    def _find_candidate_file(self, title: str, *, layer: MemoryLayer) -> Path | None:
        marker = f"## {title}".strip()
        for path in sorted(self._candidate_dir(layer).glob("*.md")):
            if marker in path.read_text(encoding="utf-8"):
                return path
        return None

    def _extract_candidate_block(self, content: str) -> str:
        marker = content.find("## ")
        if marker < 0:
            return content.strip() + "\n"
        return content[marker:].strip() + "\n"

    def _normalize_candidate_for_promotion(self, block: str) -> str:
        fields = parse_fields(block)
        lines: list[str] = []
        for raw_line in block.strip().splitlines():
            stripped = raw_line.strip()
            if not stripped.startswith("- "):
                lines.append(stripped)
                continue
            key = stripped[2:].split(":", 1)[0].strip().lower()
            if key == "status":
                lines.append("- Status: active")
            elif key == "validity":
                lines.append("- Validity: derived")
            else:
                lines.append(stripped)
        if "status" not in fields:
            lines.append("- Status: active")
        if "validity" not in fields:
            lines.append("- Validity: derived")
        return "\n".join(lines).strip() + "\n"

    def _move_candidate_to_quarantine(
        self,
        candidate_file: Path,
        *,
        layer: MemoryLayer,
        reason: str,
        source: str,
    ) -> None:
        block = self._extract_candidate_block(candidate_file.read_text(encoding="utf-8"))
        self._quarantine_block(
            block,
            layer=layer,
            reason=reason,
            source=source,
        )
        candidate_file.unlink(missing_ok=True)

    def _estimate_block_tokens(self, block: str) -> int:
        from xcode.agent.compaction import estimate_tokens

        return estimate_tokens(block)

    def _memory_id(self, layer: str, title: str) -> str:
        return build_memory_id(layer=layer, title=title)

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 3)

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
