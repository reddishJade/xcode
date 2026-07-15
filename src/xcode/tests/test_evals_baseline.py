"""版本化 Eval 基线摘要的一致性测试。"""

import json
from pathlib import Path

import pytest


def test_phase2_baseline_snapshot_is_internally_consistent() -> None:
    repository = Path(__file__).parents[3]
    path = (
        repository
        / "evals/baselines/phase2-baseline-full-deepseek-v4-flash-20260715a.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["xcode_dirty"] is False
    assert payload["task_count"] == 10
    assert payload["repetitions"] == 2
    assert payload["declared_trials"] == 20
    assert payload["observed_trials"] == payload["declared_trials"]
    assert payload["missing_trials"] == 0
    assert payload["valid_trials"] + payload["excluded_trials"] == 20
    assert payload["success_rate"] == pytest.approx(
        payload["successes"] / payload["valid_trials"]
    )

    outcomes = payload["task_outcomes"]
    assert len(outcomes) == payload["task_count"]
    assert sum(item["valid"] for item in outcomes.values()) == payload["valid_trials"]
    assert sum(item["successes"] for item in outcomes.values()) == payload["successes"]
    assert (
        sum(item["excluded"] for item in outcomes.values())
        == payload["excluded_trials"]
    )
    usage = payload["usage"]
    assert usage["tokens_per_success"] == pytest.approx(
        usage["input_tokens"] / payload["successes"]
    )
    assert usage["tool_calls_per_success"] == pytest.approx(
        usage["tool_calls"] / payload["successes"]
    )
    assert all(len(digest) == 64 for digest in payload["artifact_hashes"].values())

    serialized = path.read_text(encoding="utf-8").lower()
    assert "api_key" not in serialized
    assert "bearer " not in serialized
