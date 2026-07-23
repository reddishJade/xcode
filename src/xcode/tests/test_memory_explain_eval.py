from __future__ import annotations

import json
from pathlib import Path

import pytest

from xcode.harness.memory import (
    MemoryEvalCase,
    MemoryExclusionReason,
    MemoryManager,
    MemoryRetrievalContext,
    evaluate_case_file,
    evaluate_memory_cases,
)


def _block(
    title: str,
    *,
    memory_id: str,
    context: str = "provider timeout retry",
    status: str = "active",
    validity: str = "verified",
    confidence: float = 0.9,
    files: str = "src/provider.py",
    symbols: str = "ProviderClient",
    scope: str = "providers",
    solution: str = "Retry once with bounded backoff.",
) -> str:
    return (
        f"## {title}\n"
        f"- Memory-ID: {memory_id}\n"
        f"- Context/Query: {context}\n"
        f"- Solution: {solution}\n"
        f"- Files: {files}\n"
        f"- Related-Symbols: {symbols}\n"
        f"- Scope: {scope}\n"
        "- Takeaways: Reuse only in the matching context.\n"
        f"- Status: {status}\n"
        f"- Validity: {validity}\n"
        f"- Confidence: {confidence}\n"
    )


def _manager(
    tmp_path: Path,
    project: tuple[str, ...],
    user: tuple[str, ...] = (),
    *,
    min_retrieval_score: float = 0.0,
) -> MemoryManager:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "MEMORY.md").write_text("\n".join(project), encoding="utf-8")
    user_file = tmp_path / "user" / "MEMORY.md"
    user_file.parent.mkdir()
    user_file.write_text("\n".join(user), encoding="utf-8")
    return MemoryManager(
        tmp_path,
        user_memory_file=user_file,
        min_retrieval_score=min_retrieval_score,
    )


def test_explain_matches_real_prompt_retrieval_and_is_strictly_read_only(
    tmp_path: Path,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block("Verified", memory_id="mem_verified"),
            _block(
                "Quarantined",
                memory_id="mem_quarantined",
                status="needs_review",
                validity="contradicted",
            ),
        ),
    )
    before = (tmp_path / "MEMORY.md").read_bytes()
    trace = manager.explain_memory_retrieval(
        "provider timeout",
        limit=5,
        retrieval_context=MemoryRetrievalContext(query="provider timeout"),
        max_tokens=1200,
    )
    actual = manager.search_memory_records(
        "provider timeout",
        limit=5,
        source="prompt",
        track_usage=False,
        retrieval_context=MemoryRetrievalContext(query="provider timeout"),
    )

    assert [item.memory_id for item in trace.injected] == [
        record.memory_id for record in actual
    ]
    assert (tmp_path / "MEMORY.md").read_bytes() == before
    assert not manager.lru_file.exists()
    assert manager._session_usage == {}
    persisted = manager.read_memory_records()
    assert all(record.retrieval_count == 0 for record in persisted)
    assert all(record.injection_count == 0 for record in persisted)


@pytest.mark.parametrize(
    ("status", "validity", "reason"),
    [
        ("needs_review", "verified", MemoryExclusionReason.LIFECYCLE_STATUS),
        ("superseded", "verified", MemoryExclusionReason.LIFECYCLE_STATUS),
        ("deprecated", "verified", MemoryExclusionReason.LIFECYCLE_STATUS),
        ("obsolete", "verified", MemoryExclusionReason.LIFECYCLE_STATUS),
        ("active", "needs_review", MemoryExclusionReason.LIFECYCLE_VALIDITY),
        ("active", "corrected", MemoryExclusionReason.LIFECYCLE_VALIDITY),
        ("active", "contradicted", MemoryExclusionReason.LIFECYCLE_VALIDITY),
    ],
)
def test_explain_has_stable_lifecycle_reason_codes(
    tmp_path: Path,
    status: str,
    validity: str,
    reason: MemoryExclusionReason,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block(
                "Isolated",
                memory_id="mem_isolated",
                status=status,
                validity=validity,
            ),
        ),
    )

    decision = manager.explain_memory_retrieval("provider timeout").candidates[0]

    assert decision.decision == "excluded"
    assert decision.reason is reason


def test_score_breakdown_recomputes_final_score_and_context_matches(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, (_block("Scoped", memory_id="mem_scoped"),))
    trace = manager.explain_memory_retrieval(
        "provider timeout",
        retrieval_context=MemoryRetrievalContext(
            query="provider timeout",
            scope="providers",
            current_file="src/provider.py",
            symbols=("ProviderClient",),
        ),
    )
    decision = trace.candidates[0]

    assert decision.file_match
    assert decision.symbol_match
    assert decision.scope_match
    assert decision.score.final_score == pytest.approx(
        decision.score.fused_score * decision.score.multiplier_product,
        abs=1e-6,
    )


def test_budget_rejection_is_stable_and_never_skips_the_top_record(
    tmp_path: Path,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block(
                "Oversized",
                memory_id="mem_big",
                solution="X" * 1000,
            ),
            _block("Smaller", memory_id="mem_small", context="provider retry"),
        ),
    )

    trace = manager.explain_memory_retrieval(
        "provider timeout retry", limit=2, max_tokens=1
    )

    assert trace.used_tokens == 0
    assert all(item.decision == "budget_rejected" for item in trace.candidates[:2])
    assert all(
        item.reason is MemoryExclusionReason.BUDGET_EXCEEDED
        for item in trace.candidates[:2]
    )


def test_exact_id_audits_isolated_record_without_injection(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        (
            _block(
                "Old",
                memory_id="mem_old",
                status="needs_review",
                validity="contradicted",
            ),
        ),
    )

    trace = manager.explain_memory_retrieval("mem_old")

    assert trace.exact_id_query
    assert trace.candidates[0].reason is MemoryExclusionReason.EXACT_ID_AUDIT
    assert trace.injected == ()


@pytest.mark.parametrize(
    ("status", "validity", "expected_injected"),
    [
        ("active", "verified", True),
        ("needs_review", "verified", False),
        ("active", "contradicted", False),
        ("superseded", "verified", False),
    ],
)
def test_prompt_exact_id_matches_explain_lifecycle_decision(
    tmp_path: Path,
    status: str,
    validity: str,
    expected_injected: bool,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block(
                "Exact lifecycle",
                memory_id="mem_exact",
                status=status,
                validity=validity,
            ),
        ),
    )
    context = MemoryRetrievalContext(query="mem_exact")

    production = manager.search_memory_records(
        "mem_exact",
        source="prompt",
        track_usage=False,
        retrieval_context=context,
    )
    trace = manager.explain_memory_retrieval(
        "mem_exact",
        retrieval_context=context,
    )
    explicit_audit = manager.search_memory_records(
        "mem_exact",
        source="api",
        track_usage=False,
    )

    assert bool(production) is expected_injected
    assert bool(trace.injected) is expected_injected
    assert [record.memory_id for record in explicit_audit] == ["mem_exact"]
    if expected_injected:
        assert trace.candidates[0].reason is MemoryExclusionReason.INJECTED
    else:
        assert trace.candidates[0].reason is MemoryExclusionReason.EXACT_ID_AUDIT


def test_layers_with_same_title_remain_separately_explainable(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        (_block("Shared", memory_id="mem_project"),),
        (_block("Shared", memory_id="mem_user"),),
    )

    project = manager.explain_memory_retrieval("provider timeout", layer="project")
    user = manager.explain_memory_retrieval("provider timeout", layer="user")

    assert {item.memory_id for item in project.candidates} == {"mem_project"}
    assert {item.memory_id for item in user.candidates} == {"mem_user"}


def test_eval_metrics_json_and_failure_diagnostics_are_stable(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        (_block("English retry", memory_id="mem_en"),),
    )
    passing = MemoryEvalCase(
        case_id="english",
        query="provider timeout",
        context=MemoryRetrievalContext(query="provider timeout"),
        expected=("mem_en",),
    )
    first = evaluate_memory_cases(manager, [passing])
    second = evaluate_memory_cases(manager, [passing])

    assert first.passed
    assert first.recall_at_k == 1.0
    assert first.mrr == 1.0
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json())["schema_version"] == 1

    failing = evaluate_memory_cases(
        manager,
        [
            MemoryEvalCase(
                case_id="diagnostic",
                query="provider timeout",
                context=MemoryRetrievalContext(query="provider timeout"),
                expected=("missing",),
            )
        ],
    )
    assert not failing.passed
    assert failing.failures[0].actual
    assert failing.failures[0].explain[0]["memory_id"] == "mem_en"
    assert "missing expected" in failing.failures[0].reasons[0]


def test_default_bilingual_eval_is_a_passing_quality_gate(tmp_path: Path) -> None:
    manager = _manager(tmp_path, ())
    case_file = Path(__file__).parents[3] / "docs" / "memory_eval.yaml"

    report = evaluate_case_file(manager, case_file)

    assert report.passed
    assert report.case_count == 20
    assert report.recall_at_k >= report.recall_threshold
    assert report.mrr >= report.mrr_threshold
    assert report.forbidden_hit_count == 0
    assert report.lifecycle_safety_violation_count == 0
    assert report.budget_violation_count == 0
    assert report.deterministic_order_violation_count == 0


def test_inline_eval_inherits_embedding_and_changes_semantic_ranking(
    tmp_path: Path,
) -> None:
    def embed(text: str) -> list[float]:
        if text == "alpha beta gamma" or "Semantic match" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    lexical_block = _block(
        "Lexical match",
        memory_id="mem_lexical",
        context="alpha beta gamma",
    )
    semantic_block = _block(
        "Semantic match",
        memory_id="mem_semantic",
        context="alpha beta gamma",
    )
    case = MemoryEvalCase(
        case_id="semantic-inline",
        query="alpha beta gamma",
        context=MemoryRetrievalContext(query="alpha beta gamma"),
        expected=("mem_semantic",),
        max_rank={"mem_semantic": 1},
        project_memory=(lexical_block, semantic_block),
    )
    semantic_manager = _manager(tmp_path / "semantic", ())
    semantic_manager.embedding_fn = embed
    lexical_manager = _manager(tmp_path / "lexical", ())

    semantic_report = evaluate_memory_cases(semantic_manager, [case])
    lexical_report = evaluate_memory_cases(lexical_manager, [case])

    assert semantic_report.passed
    assert not lexical_report.passed
    assert "not ranked at or above 1" in lexical_report.failures[0].reasons[0]


def test_aggregate_metrics_do_not_retain_sensitive_text_or_paths(
    tmp_path: Path,
) -> None:
    secret_query = "SECRET-TOKEN provider timeout"
    manager = _manager(tmp_path, (_block("Safe", memory_id="mem_safe"),))
    manager.explain_memory_retrieval(secret_query)

    serialized = json.dumps(manager.retrieval_metrics.snapshot())

    assert secret_query not in serialized
    assert str(tmp_path) not in serialized
    assert "Retry once with bounded backoff" not in serialized
    assert manager.retrieval_metrics.snapshot()["retrieval_count"] == 1
