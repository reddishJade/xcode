"""只依赖结构化 Trial 声明和结果的离线 Experiment 聚合。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb

from .schema import (
    Experiment,
    ExperimentSummary,
    ResourceUsage,
    Trial,
    TrialMetric,
    TrialResult,
    UsageAggregate,
    VariantSummary,
)


@dataclass(frozen=True)
class TrialRecord:
    """经过 artifact 完整性检查后加载的 Trial 声明与结果。"""

    trial: Trial
    result: TrialResult


def aggregate_experiment(
    experiment: Experiment,
    records: tuple[TrialRecord, ...],
    *,
    pass_k: int | None = None,
) -> ExperimentSummary:
    """聚合有效能力结果，并把所有观察到的 Trial 成本计入效率。"""
    k = pass_k or experiment.repetitions
    if k < 1:
        raise ValueError("pass_k must be at least 1")
    _validate_records(experiment, records)
    by_variant: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in records:
        by_variant[record.trial.variant.variant_id].append(record)

    summaries = tuple(
        _summarize_variant(
            variant_id=variant.variant_id,
            task_count=len(experiment.task_ids),
            repetitions=experiment.repetitions,
            records=by_variant[variant.variant_id],
            pass_k=k,
        )
        for variant in experiment.variants
    )
    metrics = tuple(
        _trial_metric(record)
        for record in sorted(records, key=lambda item: item.trial.trial_id)
    )
    return ExperimentSummary(
        experiment_id=experiment.experiment_id,
        dataset_version=experiment.dataset_version,
        task_ids=experiment.task_ids,
        repetitions=experiment.repetitions,
        variants=summaries,
        efficient_variant_ids=_efficiency_frontier(summaries),
        trials=metrics,
        formulas={
            "success_rate": "successes / valid_trials",
            "pass_at_k": "mean(1 - C(n-c,k) / C(n,k)) for tasks with n >= k",
            "pass_power_k": "mean(C(c,k) / C(n,k)) for tasks with n >= k",
            "unit_cost": "sum(resource over all observed trials) / successes",
            "exclusions": "invalid trials grouped by error_category",
        },
    )


def _validate_records(
    experiment: Experiment,
    records: tuple[TrialRecord, ...],
) -> None:
    expected_variants = {variant.variant_id: variant for variant in experiment.variants}
    seen: set[str] = set()
    for record in records:
        trial = record.trial
        if trial.trial_id in seen:
            raise ValueError(f"duplicate trial artifact: {trial.trial_id}")
        seen.add(trial.trial_id)
        if record.result.trial_id != trial.trial_id:
            raise ValueError(f"trial/result identity mismatch: {trial.trial_id}")
        if trial.experiment_id != experiment.experiment_id:
            raise ValueError(f"foreign experiment trial: {trial.trial_id}")
        if trial.dataset_version != experiment.dataset_version:
            raise ValueError(f"foreign dataset trial: {trial.trial_id}")
        if trial.task_id not in experiment.task_ids:
            raise ValueError(f"unselected task trial: {trial.trial_id}")
        expected_variant = expected_variants.get(trial.variant.variant_id)
        if expected_variant is None:
            raise ValueError(f"undeclared variant trial: {trial.trial_id}")
        if trial.variant != expected_variant or trial.model != experiment.model:
            raise ValueError(f"comparison configuration mismatch: {trial.trial_id}")
        if not 0 <= trial.repetition < experiment.repetitions:
            raise ValueError(f"out-of-range repetition: {trial.trial_id}")
        expected_id = (
            f"{experiment.experiment_id}.{trial.task_id}."
            f"{trial.variant.variant_id}.r{trial.repetition}"
        )
        if trial.trial_id != expected_id:
            raise ValueError(f"noncanonical trial identity: {trial.trial_id}")


def _summarize_variant(
    *,
    variant_id: str,
    task_count: int,
    repetitions: int,
    records: list[TrialRecord],
    pass_k: int,
) -> VariantSummary:
    valid = [record for record in records if record.result.valid_trial]
    successes = sum(record.result.success for record in valid)
    by_task: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in valid:
        by_task[record.trial.task_id].append(record)
    pass_at_values: list[float] = []
    pass_power_values: list[float] = []
    for task_records in by_task.values():
        n = len(task_records)
        if n < pass_k:
            continue
        c = sum(record.result.success for record in task_records)
        pass_at_values.append(1 - comb(n - c, pass_k) / comb(n, pass_k))
        pass_power_values.append(
            comb(c, pass_k) / comb(n, pass_k) if c >= pass_k else 0.0
        )

    exclusions = Counter(
        record.result.error_category
        for record in records
        if not record.result.valid_trial and record.result.error_category is not None
    )
    return VariantSummary(
        variant_id=variant_id,
        declared_trials=task_count * repetitions,
        observed_trials=len(records),
        missing_trials=task_count * repetitions - len(records),
        valid_trials=len(valid),
        excluded_trials=len(records) - len(valid),
        successes=successes,
        success_rate=_rate(successes, len(valid)),
        pass_k=pass_k,
        pass_at_k=_mean(pass_at_values),
        pass_power_k=_mean(pass_power_values),
        pass_k_eligible_tasks=len(pass_at_values),
        resolved_rate=_verifier_rate(valid, "resolved"),
        regression_free_rate=_verifier_rate(valid, "regression_free"),
        policy_clean_rate=_verifier_rate(valid, "policy_clean"),
        exclusions=dict(exclusions),
        usage=_aggregate_usage(
            tuple(record.result.usage for record in records),
            successes=successes,
        ),
    )


def _trial_metric(record: TrialRecord) -> TrialMetric:
    verifier = record.result.verifier
    return TrialMetric(
        trial_id=record.trial.trial_id,
        task_id=record.trial.task_id,
        variant_id=record.trial.variant.variant_id,
        repetition=record.trial.repetition,
        valid_trial=record.result.valid_trial,
        success=record.result.success,
        error_category=record.result.error_category,
        resolved=verifier.resolved if verifier is not None else None,
        regression_free=verifier.regression_free if verifier is not None else None,
        policy_clean=verifier.policy_clean if verifier is not None else None,
        usage=record.result.usage,
    )


def _aggregate_usage(
    usages: tuple[ResourceUsage, ...],
    *,
    successes: int,
) -> UsageAggregate:
    wall_time = sum(usage.wall_time_seconds for usage in usages)
    model_calls = sum(usage.model_calls for usage in usages)
    tool_calls = sum(usage.tool_calls for usage in usages)
    input_tokens = _optional_int_sum(tuple(usage.input_tokens for usage in usages))
    output_tokens = _optional_int_sum(tuple(usage.output_tokens for usage in usages))
    costs = _optional_float_sum(tuple(usage.cost_usd for usage in usages))
    return UsageAggregate(
        wall_time_seconds=wall_time,
        model_calls=model_calls,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=costs,
        tokens_per_success=(
            input_tokens / successes if input_tokens is not None and successes else None
        ),
        tool_calls_per_success=tool_calls / successes if successes else None,
        time_per_success=wall_time / successes if successes else None,
        cost_per_success=costs / successes if costs is not None and successes else None,
    )


def _optional_int_sum(values: tuple[int | None, ...]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _optional_float_sum(values: tuple[float | None, ...]) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _verifier_rate(records: list[TrialRecord], field: str) -> float | None:
    passed = sum(
        bool(getattr(record.result.verifier, field))
        for record in records
        if record.result.verifier is not None
    )
    return _rate(passed, len(records))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _efficiency_frontier(summaries: tuple[VariantSummary, ...]) -> tuple[str, ...]:
    eligible = [
        summary
        for summary in summaries
        if summary.success_rate is not None and summary.usage.input_tokens is not None
    ]
    efficient: list[str] = []
    for candidate in eligible:
        dominated = any(
            _dominates(other, candidate)
            for other in eligible
            if other.variant_id != candidate.variant_id
        )
        if not dominated:
            efficient.append(candidate.variant_id)
    return tuple(efficient)


def _dominates(candidate: VariantSummary, other: VariantSummary) -> bool:
    assert candidate.success_rate is not None
    assert other.success_rate is not None
    assert candidate.usage.input_tokens is not None
    assert other.usage.input_tokens is not None
    no_worse = (
        candidate.success_rate >= other.success_rate
        and candidate.usage.input_tokens <= other.usage.input_tokens
        and candidate.usage.wall_time_seconds <= other.usage.wall_time_seconds
    )
    strictly_better = (
        candidate.success_rate > other.success_rate
        or candidate.usage.input_tokens < other.usage.input_tokens
        or candidate.usage.wall_time_seconds < other.usage.wall_time_seconds
    )
    return no_worse and strictly_better
