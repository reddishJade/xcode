"""确定性的 Memory 离线评测、指标与质量门禁。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, cast

import yaml

from .manager import MemoryLayerFilter, MemoryManager, MemoryRetrievalContext
from .retrieval import MemoryCandidateDecision, MemoryRetrievalTrace


@dataclass(frozen=True)
class MemoryEvalCase:
    """一个可版本控制的检索评测案例。"""

    case_id: str
    query: str
    context: MemoryRetrievalContext
    layer: MemoryLayerFilter = "all"
    top_k: int = 3
    expected: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    require_injection: bool = True
    token_budget: int | None = 1200
    min_rank: dict[str, int] | None = None
    max_rank: dict[str, int] | None = None
    project_memory: tuple[str, ...] = ()
    user_memory: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryEvalFailure:
    """包含实际结果和关键 explain 信息的失败诊断。"""

    case_id: str
    reasons: tuple[str, ...]
    expected: tuple[str, ...]
    actual: tuple[str, ...]
    explain: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MemoryEvalReport:
    """稳定、可供 CI 消费的评测报告。"""

    schema_version: int
    case_count: int
    recall_at_k: float
    mrr: float
    forbidden_hit_count: int
    lifecycle_safety_violation_count: int
    budget_violation_count: int
    deterministic_order_violation_count: int
    recall_threshold: float
    mrr_threshold: float
    passed: bool
    failures: tuple[MemoryEvalFailure, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为字段顺序稳定的 JSON 对象。"""
        return asdict(self)

    def to_json(self) -> str:
        """输出稳定 JSON，便于 CI 与可视化消费。"""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def render(self) -> str:
        """渲染可诊断的文本报告。"""
        lines = [
            f"Memory eval: {'PASS' if self.passed else 'FAIL'} ({self.case_count} cases)",
            f"- Recall@K: {self.recall_at_k:.3f} (>= {self.recall_threshold:.3f})",
            f"- MRR: {self.mrr:.3f} (>= {self.mrr_threshold:.3f})",
            f"- forbidden hits: {self.forbidden_hit_count}",
            f"- lifecycle safety violations: {self.lifecycle_safety_violation_count}",
            f"- budget violations: {self.budget_violation_count}",
            (
                "- deterministic-order violations: "
                f"{self.deterministic_order_violation_count}"
            ),
        ]
        for failure in self.failures:
            lines.append(
                f"- case {failure.case_id}: {'; '.join(failure.reasons)}; "
                f"expected={list(failure.expected)!r}; actual={list(failure.actual)!r}"
            )
        return "\n".join(lines)


def load_memory_eval_cases(path: Path) -> list[MemoryEvalCase]:
    """从 JSON 或 YAML 加载版本 1 评测案例。"""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("memory eval case file must use version: 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("memory eval case file must contain a cases list")
    return [_parse_case(item, index) for index, item in enumerate(raw_cases)]


def _parse_case(value: object, index: int) -> MemoryEvalCase:
    if not isinstance(value, dict):
        raise ValueError(f"case {index} must be an object")
    item = cast(dict[str, Any], value)
    query = str(item.get("query", "")).strip()
    if not query:
        raise ValueError(f"case {index} query is required")
    raw_context = item.get("context") or {}
    if not isinstance(raw_context, dict):
        raise ValueError(f"case {index} context must be an object")
    context = cast(dict[str, Any], raw_context)
    layer = str(item.get("layer", "all"))
    if layer not in {"all", "project", "user"}:
        raise ValueError(f"case {index} has invalid layer")
    return MemoryEvalCase(
        case_id=str(item.get("id") or f"case-{index + 1}"),
        query=query,
        context=MemoryRetrievalContext(
            query=query,
            scope=_optional_string(context.get("scope")),
            current_file=_optional_string(context.get("current_file")),
            symbols=_string_tuple(context.get("symbols")),
            error_messages=_string_tuple(context.get("error_messages")),
            task_phase=_optional_string(context.get("task_phase")),
            modules=_string_tuple(context.get("modules")),
            recent_files=_string_tuple(context.get("recent_files")),
        ),
        layer=cast(MemoryLayerFilter, layer),
        top_k=max(0, int(item.get("top_k", 3))),
        expected=_string_tuple(item.get("expected")),
        forbidden=_string_tuple(item.get("forbidden")),
        require_injection=bool(item.get("require_injection", True)),
        token_budget=(
            None if item.get("token_budget") is None else int(item["token_budget"])
        ),
        min_rank=_rank_map(item.get("min_rank")),
        max_rank=_rank_map(item.get("max_rank")),
        project_memory=_string_tuple(item.get("project_memory")),
        user_memory=_string_tuple(item.get("user_memory")),
    )


def evaluate_memory_cases(
    manager: MemoryManager,
    cases: list[MemoryEvalCase],
    *,
    recall_threshold: float = 0.8,
    mrr_threshold: float = 0.7,
) -> MemoryEvalReport:
    """只读执行案例两次，并计算检索质量与安全指标。"""
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    forbidden_hits = 0
    safety_violations = 0
    budget_violations = 0
    deterministic_violations = 0
    failures: list[MemoryEvalFailure] = []
    for case in cases:
        with _case_manager(manager, case) as case_manager:
            first = case_manager.explain_memory_retrieval(
                case.query,
                limit=case.top_k,
                layer=case.layer,
                retrieval_context=case.context,
                max_tokens=case.token_budget,
            )
            second = case_manager.explain_memory_retrieval(
                case.query,
                limit=case.top_k,
                layer=case.layer,
                retrieval_context=case.context,
                max_tokens=case.token_budget,
            )
        pool = first.injected if case.require_injection else first.candidates
        actual = tuple(_decision_label(item) for item in pool)
        matched = [
            expected
            for expected in case.expected
            if any(_matches(item, expected) for item in pool)
        ]
        recalls.append(len(matched) / len(case.expected) if case.expected else 1.0)
        ranks = [
            index
            for index, item in enumerate(pool, start=1)
            if any(_matches(item, expected) for expected in case.expected)
        ]
        reciprocal_ranks.append(
            1.0 if not case.expected else (1.0 / min(ranks) if ranks else 0.0)
        )
        case_forbidden = sum(
            1
            for item in first.injected
            if any(_matches(item, forbidden) for forbidden in case.forbidden)
        )
        forbidden_hits += case_forbidden
        case_safety = sum(
            1
            for item in first.injected
            if item.status in {"needs_review", "superseded", "deprecated", "obsolete"}
            or item.validity in {"needs_review", "corrected", "contradicted"}
        )
        safety_violations += case_safety
        case_budget = int(
            case.token_budget is not None
            and first.used_tokens > max(case.token_budget, 0)
        )
        budget_violations += case_budget
        first_order = tuple(
            (item.memory_id, item.decision, item.reason.value)
            for item in first.candidates
        )
        second_order = tuple(
            (item.memory_id, item.decision, item.reason.value)
            for item in second.candidates
        )
        case_deterministic = int(first_order != second_order)
        deterministic_violations += case_deterministic
        reasons = _case_failure_reasons(
            case,
            pool,
            matched,
            case_forbidden,
            case_safety,
            case_budget,
            case_deterministic,
        )
        if reasons:
            failures.append(
                MemoryEvalFailure(
                    case_id=case.case_id,
                    reasons=tuple(reasons),
                    expected=case.expected,
                    actual=actual,
                    explain=_diagnostic_explain(first),
                )
            )
    recall = sum(recalls) / max(len(recalls), 1)
    mrr = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
    passed = (
        recall >= recall_threshold
        and mrr >= mrr_threshold
        and forbidden_hits == 0
        and safety_violations == 0
        and budget_violations == 0
        and deterministic_violations == 0
        and not failures
    )
    return MemoryEvalReport(
        schema_version=1,
        case_count=len(cases),
        recall_at_k=round(recall, 6),
        mrr=round(mrr, 6),
        forbidden_hit_count=forbidden_hits,
        lifecycle_safety_violation_count=safety_violations,
        budget_violation_count=budget_violations,
        deterministic_order_violation_count=deterministic_violations,
        recall_threshold=recall_threshold,
        mrr_threshold=mrr_threshold,
        passed=passed,
        failures=tuple(failures),
    )


def _case_failure_reasons(
    case: MemoryEvalCase,
    pool: tuple[MemoryCandidateDecision, ...],
    matched: list[str],
    forbidden_hits: int,
    safety: int,
    budget: int,
    deterministic: int,
) -> list[str]:
    reasons: list[str] = []
    missing = [item for item in case.expected if item not in matched]
    if missing:
        reasons.append(f"missing expected: {missing!r}")
    if forbidden_hits:
        reasons.append(f"forbidden hits: {forbidden_hits}")
    if safety:
        reasons.append(f"lifecycle safety violations: {safety}")
    if budget:
        reasons.append("token budget exceeded")
    if deterministic:
        reasons.append("candidate order changed between identical runs")
    ranks = {
        key: index
        for index, item in enumerate(pool, start=1)
        for key in (item.memory_id, item.title)
    }
    for label, minimum in (case.min_rank or {}).items():
        if label in ranks and ranks[label] < minimum:
            reasons.append(f"{label!r} rank {ranks[label]} is below minimum {minimum}")
    for label, maximum in (case.max_rank or {}).items():
        if label not in ranks or ranks[label] > maximum:
            reasons.append(f"{label!r} is not ranked at or above {maximum}")
    return reasons


def _diagnostic_explain(trace: MemoryRetrievalTrace) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "memory_id": item.memory_id,
            "title": item.title,
            "layer": item.layer,
            "rank": item.rank,
            "decision": item.decision,
            "reason": item.reason.value,
            "lexical_score": item.score.lexical_score,
            "semantic_score": item.score.semantic_score,
            "final_score": item.score.final_score,
            "token_count": item.token_count,
        }
        for item in trace.candidates
    )


def _matches(item: MemoryCandidateDecision, expected: str) -> bool:
    normalized = expected.casefold()
    return normalized in {item.memory_id.casefold(), item.title.casefold()}


def _decision_label(item: MemoryCandidateDecision) -> str:
    return f"{item.layer}:{item.memory_id}:{item.title}"


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError("expected a string list")
    return tuple(str(item) for item in value)


def _rank_map(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("rank constraint must be an object")
    return {str(key): int(rank) for key, rank in value.items()}


def evaluate_case_file(manager: MemoryManager, path: Path) -> MemoryEvalReport:
    """加载并评测案例文件，供 CLI/pytest/CI 共用。"""
    return evaluate_memory_cases(manager, load_memory_eval_cases(path))


class _case_manager:
    """为带内联语料的案例创建隔离的临时 MemoryManager。"""

    def __init__(self, base: MemoryManager, case: MemoryEvalCase) -> None:
        self.base = base
        self.case = case
        self.temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> MemoryManager:
        if not self.case.project_memory and not self.case.user_memory:
            return self.base
        self.temporary = tempfile.TemporaryDirectory(prefix="xcode-memory-eval-")
        root = Path(self.temporary.name)
        project_file = root / "MEMORY.md"
        user_file = root / "user" / "MEMORY.md"
        user_file.parent.mkdir(parents=True)
        project_file.write_text("\n".join(self.case.project_memory), encoding="utf-8")
        user_file.write_text("\n".join(self.case.user_memory), encoding="utf-8")
        return MemoryManager(
            root,
            user_memory_file=user_file,
            min_retrieval_score=self.base.min_retrieval_score,
            min_confidence=self.base.min_confidence,
            rerank_policy=self.base.rerank_policy,
            lifecycle_policy=self.base.lifecycle_policy,
            embedding_fn=self.base.embedding_fn,
        )

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()
