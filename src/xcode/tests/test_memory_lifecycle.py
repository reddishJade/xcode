"""长期记忆生命周期、来源和召回门测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from xcode.harness.memory import (
    MemoryEvidence,
    MemoryJudgeResult,
    MemoryLifecyclePolicy,
    MemoryManager,
    MemoryRetrievalContext,
    build_memory_tools,
)


def _block(
    title: str,
    solution: str,
    *,
    status: str | None = None,
    validity: str | None = None,
    confidence: float | None = None,
    memory_id: str | None = None,
) -> str:
    lines = [
        f"## {title}",
        "- Context/Query: provider timeout lifecycle behavior",
        f"- Solution: {solution}",
        "- Files: src/provider.py",
        "- Takeaways: Reuse the scoped provider decision across sessions.",
    ]
    for key, value in (
        ("Memory-ID", memory_id),
        ("Status", status),
        ("Validity", validity),
        ("Confidence", str(confidence) if confidence is not None else None),
    ):
        if value is not None:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _manager(tmp_path: Path, blocks: tuple[str, ...] = ()) -> MemoryManager:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "MEMORY.md").write_text("\n".join(blocks), encoding="utf-8")
    user_file = tmp_path / "user" / "MEMORY.md"
    user_file.parent.mkdir(exist_ok=True)
    user_file.write_text("", encoding="utf-8")
    return MemoryManager(
        tmp_path,
        user_memory_file=user_file,
        min_retrieval_score=0.0,
        lifecycle_policy=MemoryLifecyclePolicy(
            candidate_promotion_successes=2,
            verification_successes=2,
        ),
    )


def test_legacy_and_explicit_add_defaults_are_compatible(tmp_path: Path) -> None:
    manager = _manager(tmp_path, (_block("Legacy", "Retry once."),))
    legacy = manager.read_memory_records(layer="project")[0]
    assert legacy.status == "active"
    assert legacy.validity == "unknown"
    assert legacy.evidence == ()

    assert manager.add_memory_block(
        _block("Explicit", "Use bounded backoff."),
        source="repl",
    )
    explicit = next(
        record
        for record in manager.read_memory_records(layer="project")
        if record.title == "Explicit"
    )
    assert explicit.status == "active"
    assert explicit.validity == "user_confirmed"
    assert explicit.source_session is None
    assert explicit.evidence == (MemoryEvidence("user", "explicit-confirmation"),)


def test_explicit_metadata_is_not_overridden(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    assert manager.add_memory_block(
        _block("Tentative", "Try the scoped retry."),
        source="repl",
        status="candidate",
        validity="derived",
        confidence=0.4,
    )
    record = manager.read_memory_records(layer="project")[0]
    assert (record.status, record.validity, record.confidence_value) == (
        "candidate",
        "derived",
        pytest.approx(0.4),
    )


@pytest.mark.parametrize(
    ("confidence", "scope", "files", "expected_status"),
    [
        (0.4, "providers", ("src/provider.py",), "candidate"),
        (0.9, "providers", ("src/provider.py",), "active"),
    ],
)
def test_consolidation_uses_deterministic_quality_split_and_provenance(
    tmp_path: Path,
    confidence: float,
    scope: str,
    files: tuple[str, ...],
    expected_status: str,
) -> None:
    def judge(_text: str) -> list[MemoryJudgeResult]:
        return [
            MemoryJudgeResult(
                is_worth_remembering=True,
                confidence=confidence,
                scope=scope,
                related_files=files,
                suggested_title="Durable provider retry",
                suggested_context="Provider timeout policy",
                suggested_solution="Retry once with bounded backoff.",
                suggested_takeaways="Reuse only for provider timeouts.",
            )
        ]

    manager = _manager(tmp_path)
    manager.set_consolidate_judge_fn(judge)
    manager.consolidate(
        "## Goal\nFinish this temporary task\n"
        "## Progress\nHalf done\n"
        "## Key Decisions\n- Keep provider retries bounded\n"
        "## Next Steps\nRun tests",
        source_session="session-real",
        source_message="message-real",
    )

    records = manager.read_memory_records(layer="project")
    assert [record.title for record in records] == ["Durable provider retry"]
    record = records[0]
    assert (record.status, record.validity) == (expected_status, "derived")
    assert record.source_session == "session-real"
    assert record.source_message == "message-real"
    assert record.evidence == (
        MemoryEvidence("session", "session-real"),
        MemoryEvidence("message", "message-real"),
    )


def test_goal_progress_and_next_steps_never_seed_long_term_memory(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manager.consolidate(
        "## Goal\nImplement a one-off task in a temporary file\n"
        "## Progress\nCurrent turn is half done\n"
        "## Key Decisions\n(none)\n"
        "## Next Steps\nDelete the temp file"
    )
    assert manager.read_memory_records(layer="project") == []


def test_candidate_promotes_only_after_confirmed_adoption(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        (_block("Candidate", "Retry once.", status="candidate", validity="derived"),),
    )
    for _ in range(2):
        record = manager.read_memory_records(layer="project")[0]
        manager.record_adopted_records([record])
        manager.record_session_outcome("success")
    promoted = manager.read_memory_records(layer="project")[0]
    assert promoted.status == "active"
    assert promoted.validity == "verified"

    untouched_manager = _manager(
        tmp_path / "untouched",
        (_block("Candidate", "Retry once.", status="candidate", validity="derived"),),
    )
    record = untouched_manager.search_memory_records("provider timeout")[0]
    untouched_manager.record_injected_records([record])
    untouched_manager.record_session_outcome("success")
    untouched = untouched_manager.read_memory_records(layer="project")[0]
    assert untouched.status == "candidate"
    assert untouched.success_count == 0


def test_contradicted_is_not_injected_but_exact_id_remains_auditable(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, (_block("Old rule", "Retry forever."),))
    record = manager.read_memory_records(layer="project")[0]
    assert manager.record_contradiction(
        record.memory_id,
        evidence=(MemoryEvidence("test", "tests/test_retry.py"),),
    )
    assert (
        manager.search_memory_records(
            "provider timeout",
            source="prompt",
            retrieval_context=MemoryRetrievalContext(query="provider timeout"),
        )
        == []
    )
    audited = manager.search_memory_records(record.memory_id)[0]
    assert audited.validity == "contradicted"
    assert audited.status == "needs_review"


def test_active_verified_ranks_before_candidate_and_tool_labels_candidate(
    tmp_path: Path,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block("Verified", "Retry once.", validity="verified", confidence=0.9),
            _block(
                "Candidate",
                "Retry once.",
                status="candidate",
                validity="derived",
                confidence=0.9,
            ),
        ),
    )
    records = manager.search_memory_records("provider timeout", limit=2)
    assert records[0].title == "Verified"
    output = build_memory_tools(manager)[0].handler(
        {"query": "provider timeout", "limit": 2}, None
    )
    assert "status=candidate validity=derived" in output


def test_candidate_prompt_gate_requires_confidence_and_strong_scope_match(
    tmp_path: Path,
) -> None:
    manager = _manager(
        tmp_path,
        (
            _block(
                "Weak candidate",
                "Retry once.",
                status="candidate",
                validity="derived",
                confidence=0.5,
            ),
            _block(
                "Scoped candidate",
                "Use bounded backoff.",
                status="candidate",
                validity="derived",
                confidence=0.9,
            ),
        ),
    )
    records = manager.search_memory_records(
        "provider timeout",
        limit=5,
        source="prompt",
        retrieval_context=MemoryRetrievalContext(
            query="provider timeout", current_file="src/provider.py"
        ),
    )
    assert [record.title for record in records] == ["Scoped candidate"]


def test_same_title_merge_conflict_and_explicit_supersedes(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    assert manager.add_memory_block(
        _block("Retry policy", "Retry once."),
        source_session="session-one",
        evidence=(MemoryEvidence("test", "test_retry"),),
    )
    original = manager.read_memory_records(layer="project")[0]
    assert manager.add_memory_block(
        _block("Retry policy", "Retry once."),
        source_session="session-two",
        evidence=(MemoryEvidence("commit", "abc123"),),
    )
    merged = manager.read_memory_records(layer="project")[0]
    assert merged.memory_id == original.memory_id
    assert set(merged.evidence) == {
        MemoryEvidence("test", "test_retry"),
        MemoryEvidence("commit", "abc123"),
    }
    assert merged.source_session == "session-one, session-two"

    assert manager.add_memory_block(_block("Retry policy", "Never retry."))
    records = manager.read_memory_records(layer="project")
    conflict = next(record for record in records if "(conflict " in record.title)
    assert conflict.status == "needs_review"
    assert conflict.validity == "contradicted"
    assert conflict.supersedes == (original.memory_id,)
    assert next(record for record in records if record.memory_id == original.memory_id)

    assert manager.add_memory_block(
        _block("Replacement policy", "Use circuit breaking."),
        supersedes=(original.memory_id,),
        evidence=(MemoryEvidence("user", "confirmed"),),
    )
    old = next(
        record
        for record in manager.read_memory_records(layer="project")
        if record.memory_id == original.memory_id
    )
    assert old.status == "superseded"


def test_duplicate_body_merges_new_evidence_without_cross_layer_pollution(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    assert manager.add_memory_block(
        _block("Project title", "Retry once."),
        evidence=(MemoryEvidence("test", "first"),),
    )
    assert manager.add_memory_block(
        _block("Alternate title", "Retry once."),
        evidence=(MemoryEvidence("commit", "second"),),
    )
    project = manager.read_memory_records(layer="project")
    assert len(project) == 1
    assert {item.kind for item in project[0].evidence} == {"test", "commit"}

    assert manager.add_memory_block(
        _block("Project title", "Use a user preference."),
        layer="user",
    )
    assert len(manager.read_memory_records(layer="project")) == 1
    assert len(manager.read_memory_records(layer="user")) == 1
