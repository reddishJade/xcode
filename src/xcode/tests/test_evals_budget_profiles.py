"""预算档解析契约测试。"""

import pytest

from xcode.evals.budget_profiles import (
    BudgetProfileError,
    available_budget_profiles,
    resolve_budget_profile,
)


def test_budget_profiles_are_stable_and_explicit() -> None:
    assert available_budget_profiles() == ("task", "external-20", "external-40", "external-60")
    budget = resolve_budget_profile("external-40")
    assert budget is not None
    assert budget.model_calls == 40
    assert budget.wall_time_seconds == 600


def test_task_profile_uses_task_budget() -> None:
    assert resolve_budget_profile("task") is None


def test_unknown_budget_profile_is_rejected() -> None:
    with pytest.raises(BudgetProfileError, match="unknown budget profile"):
        resolve_budget_profile("external-999")
