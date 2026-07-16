"""可复现的预算档定义。"""

from __future__ import annotations

from .schema import ResourceBudget


class BudgetProfileError(ValueError):
    """预算档标识未知或不能用于 Experiment。"""


_PROFILES: dict[str, ResourceBudget] = {
    "external-20": ResourceBudget(
        wall_time_seconds=300,
        model_calls=20,
        tool_calls=60,
        input_tokens=2_000_000,
        output_tokens=100_000,
    ),
    "external-40": ResourceBudget(
        wall_time_seconds=600,
        model_calls=40,
        tool_calls=120,
        input_tokens=4_000_000,
        output_tokens=200_000,
    ),
    "external-60": ResourceBudget(
        wall_time_seconds=900,
        model_calls=60,
        tool_calls=180,
        input_tokens=6_000_000,
        output_tokens=300_000,
    ),
}


def available_budget_profiles() -> tuple[str, ...]:
    """返回 CLI 可选择的稳定预算档标识。"""
    return ("task", *_PROFILES)


def resolve_budget_profile(profile_id: str) -> ResourceBudget | None:
    """解析预算档；`task` 表示使用任务自身声明的预算。"""
    if profile_id == "task":
        return None
    try:
        return _PROFILES[profile_id]
    except KeyError as error:
        raise BudgetProfileError(f"unknown budget profile: {profile_id}") from error
