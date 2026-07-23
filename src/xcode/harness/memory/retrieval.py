"""Memory 检索解释与聚合可观测性模型。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class MemoryExclusionReason(StrEnum):
    """稳定的检索决策原因码。"""

    INJECTED = "injected"
    EXACT_ID_AUDIT = "exact_id_audit"
    LIFECYCLE_STATUS = "lifecycle_status"
    LIFECYCLE_VALIDITY = "lifecycle_validity"
    CANDIDATE_LOW_CONFIDENCE = "candidate_low_confidence"
    CANDIDATE_CONTEXT_MISMATCH = "candidate_context_mismatch"
    CONFIDENCE_BELOW_MINIMUM = "confidence_below_minimum"
    SCORE_BELOW_MINIMUM = "score_below_minimum"
    NON_FINITE_SCORE = "non_finite_score"
    LIMIT_EXCEEDED = "limit_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"


type MemoryDecision = Literal["injected", "excluded", "budget_rejected"]


@dataclass(frozen=True)
class MemoryScoreBreakdown:
    """由真实评分流水线产生的可复算分数构成。"""

    bm25_score: float = 0.0
    lexical_score: float = 0.0
    semantic_score: float | None = None
    fused_score: float = 0.0
    status_multiplier: float = 1.0
    confidence_multiplier: float = 1.0
    validity_multiplier: float = 1.0
    utility_multiplier: float = 1.0
    negative_transfer_multiplier: float = 1.0
    scope_multiplier: float = 1.0
    freshness_multiplier: float = 1.0
    provenance_multiplier: float = 1.0
    final_score: float = 0.0

    @property
    def multiplier_product(self) -> float:
        """返回所有重排倍率的乘积。"""
        return (
            self.status_multiplier
            * self.confidence_multiplier
            * self.validity_multiplier
            * self.utility_multiplier
            * self.negative_transfer_multiplier
            * self.scope_multiplier
            * self.freshness_multiplier
            * self.provenance_multiplier
        )


@dataclass(frozen=True)
class MemoryCandidateDecision:
    """一条 Memory 候选在检索与预算流水线中的最终决策。"""

    memory_id: str
    title: str
    layer: str
    status: str
    validity: str
    score: MemoryScoreBreakdown
    scope_match: bool | None
    file_match: bool
    symbol_match: bool
    rank: int | None
    token_count: int
    decision: MemoryDecision
    reason: MemoryExclusionReason


@dataclass(frozen=True)
class MemoryRetrievalTrace:
    """一次只读检索的完整、稳定解释。"""

    query_fingerprint: str
    layer: str
    limit: int
    token_budget: int | None
    used_tokens: int
    exact_id_query: bool
    elapsed_ms: float
    candidates: tuple[MemoryCandidateDecision, ...]

    @property
    def injected(self) -> tuple[MemoryCandidateDecision, ...]:
        return tuple(item for item in self.candidates if item.decision == "injected")

    def to_dict(self) -> dict[str, Any]:
        """转换为不包含 query 或 Memory 正文的稳定 JSON 对象。"""
        return asdict(self)

    def render(self) -> str:
        """渲染适合 REPL 的精简解释。"""
        lines = [
            (
                f"Memory retrieval: {len(self.candidates)} candidates, "
                f"{len(self.injected)} injected, tokens={self.used_tokens}"
                + (f"/{self.token_budget}" if self.token_budget is not None else "")
            )
        ]
        for item in self.candidates:
            rank = f"#{item.rank}" if item.rank is not None else "-"
            semantic = (
                f" semantic={item.score.semantic_score:.3f}"
                if item.score.semantic_score is not None
                else ""
            )
            lines.append(
                f"- {rank} [{item.layer}] {item.memory_id} {item.title}: "
                f"{item.decision}/{item.reason.value}; "
                f"lexical={item.score.lexical_score:.3f}{semantic} "
                f"final={item.score.final_score:.3f} tokens={item.token_count}"
            )
        return "\n".join(lines)


@dataclass
class MemoryRetrievalMetrics:
    """仅保存枚举和数值的进程内聚合检索指标。"""

    retrieval_count: int = 0
    candidate_count: int = 0
    injected_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_budget_usage: float = 0.0
    budgeted_retrieval_count: int = 0
    exclusion_reasons: Counter[str] = field(default_factory=Counter)
    lifecycle_statuses: Counter[str] = field(default_factory=Counter)
    layer_hits: Counter[str] = field(default_factory=Counter)

    def observe(self, trace: MemoryRetrievalTrace) -> None:
        """聚合一次解释，不保留 query、正文或路径。"""
        self.retrieval_count += 1
        self.candidate_count += len(trace.candidates)
        self.injected_count += len(trace.injected)
        self.total_latency_ms += trace.elapsed_ms
        self.max_latency_ms = max(self.max_latency_ms, trace.elapsed_ms)
        if trace.token_budget is not None:
            self.budgeted_retrieval_count += 1
            if trace.token_budget > 0:
                self.total_budget_usage += trace.used_tokens / trace.token_budget
        for item in trace.candidates:
            self.lifecycle_statuses[item.status] += 1
            if item.decision != "injected":
                self.exclusion_reasons[item.reason.value] += 1
            else:
                self.layer_hits[item.layer] += 1

    def observe_search(
        self,
        *,
        candidate_statuses: tuple[str, ...],
        injected_layers: tuple[str, ...],
        exclusion_reasons: tuple[MemoryExclusionReason, ...],
        latency_ms: float,
    ) -> None:
        """聚合普通检索，不要求保存完整解释对象。"""
        self.retrieval_count += 1
        self.candidate_count += len(candidate_statuses)
        self.injected_count += len(injected_layers)
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.lifecycle_statuses.update(candidate_statuses)
        self.layer_hits.update(injected_layers)
        self.exclusion_reasons.update(reason.value for reason in exclusion_reasons)

    def observe_budget(self, *, used_tokens: int, token_budget: int) -> None:
        """聚合一次实际 prompt 预算使用率。"""
        self.budgeted_retrieval_count += 1
        if token_budget > 0:
            self.total_budget_usage += used_tokens / token_budget

    def snapshot(self) -> dict[str, Any]:
        """返回适合 telemetry 消费且不含敏感文本的稳定快照。"""
        return {
            "retrieval_count": self.retrieval_count,
            "candidate_count": self.candidate_count,
            "injected_count": self.injected_count,
            "exclusion_reasons": dict(sorted(self.exclusion_reasons.items())),
            "lifecycle_statuses": dict(sorted(self.lifecycle_statuses.items())),
            "average_latency_ms": round(
                self.total_latency_ms / max(self.retrieval_count, 1), 3
            ),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "average_budget_usage": round(
                self.total_budget_usage / max(self.budgeted_retrieval_count, 1), 6
            ),
            "layer_hits": dict(sorted(self.layer_hits.items())),
        }
