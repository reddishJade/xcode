from __future__ import annotations

import asyncio
from dataclasses import asdict
from collections.abc import Callable, Sequence
from datetime import datetime, UTC
from pathlib import Path
import hashlib
import math
import subprocess
import sys
import uuid
from typing import Any

from xcode.harness.app import XcodeApp
from xcode.harness.agent_runtime import StructuredAgentEvent

from .graders import grade_events, run_llm_judge
from .reporting import write_report_files
from .schema import EvalReport, EvalTask, GraderResult, TrialResult
from .tracing import TraceRecorder
from .validation import run_validation, validation_results_to_dict

AppFactory = Callable[[EvalTask, int], XcodeApp]


class EvalRunner:
    """面向 Agent 事件流的最小 eval pipeline。"""

    def __init__(
        self,
        tasks: Sequence[EvalTask],
        app_factory: AppFactory,
        output_dir: Path | None = None,
        trials_per_task: int = 1,
    ) -> None:
        if trials_per_task < 1:
            raise ValueError("trials_per_task must be >= 1")
        self.tasks = tuple(tasks)
        self.app_factory = app_factory
        self.run_id = _new_run_id()
        self.output_dir = (
            output_dir or Path.cwd() / ".local" / "eval_runs" / self.run_id
        )
        self.trials_per_task = trials_per_task

    def run(self) -> EvalReport:
        return _run_coro_sync(self.arun())

    async def arun(self) -> EvalReport:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total_trials = len(self.tasks) * self.trials_per_task
        trials: list[TrialResult] = []
        for task in self.tasks:
            for trial_index in range(self.trials_per_task):
                trial_no = len(trials) + 1
                print(
                    f"\r[{trial_no}/{total_trials}] {task.id} "
                    f"trial {trial_index + 1}/{self.trials_per_task} ...",
                    end="",
                    file=sys.stderr,
                )
                sys.stderr.flush()
                result = await self._run_trial(task, trial_index)
                trials.append(result)
                status = "PASS" if result.success else "FAIL"
                print(
                    f"\r[{trial_no}/{total_trials}] {task.id} "
                    f"trial {trial_index + 1}/{self.trials_per_task} {status}",
                    file=sys.stderr,
                )
        success = all(trial.success for trial in trials)
        metrics = _build_run_metrics(self.tasks, trials)
        report = EvalReport(
            run_id=self.run_id,
            success=success,
            output_dir=self.output_dir,
            trials=tuple(trials),
            metrics=metrics,
        )
        write_report_files(report)
        return report

    async def _run_trial(self, task: EvalTask, trial_index: int) -> TrialResult:
        trial_id = f"{task.id}-{trial_index + 1}"
        trace_path = self.output_dir / f"{trial_id}.jsonl"
        app = self.app_factory(task, trial_index)
        try:
            events: list[StructuredAgentEvent] = []
            answer = ""
            runtime_error: BaseException | None = None

            project_root = getattr(app.agent, "project_root", None)
            before_evidence = _collect_file_evidence(task, project_root)
            with TraceRecorder(trace_path) as trace:
                try:
                    async for event in app.aask_stream(task.prompt, mode=task.mode):
                        events.append(event)
                        trace.record(event)
                        if event.type == "final":
                            answer = event.data.answer
                except BaseException as exc:
                    runtime_error = exc
                    trace.record_error(exc)

            after_evidence = _collect_file_evidence(task, project_root)
            memory_trace = _collect_memory_trace(app)
            evidence_graders = _grade_file_evidence(
                task, before_evidence, after_evidence
            )
            validation_graders, validation_results = run_validation(task, project_root)
            graders = (
                grade_events(task, events, answer, runtime_error)
                + evidence_graders
                + validation_graders
            )

            # LLM-as-judge：当 task.llm_judge_criteria 非空时执行
            if task.llm_judge_criteria:
                judge_graders = await run_llm_judge(
                    task,
                    answer,
                    events,
                    judge_provider=app.agent.provider,
                )
                graders = graders + judge_graders

            success = all(grader.passed for grader in graders)
            tool_call_count = sum(1 for event in events if event.type == "tool_use")
            tool_error_count = sum(
                1
                for event in events
                if event.type == "tool_result"
                and getattr(event.data, "status", "ok") not in {"ok", "interrupted"}
            )
            metrics: dict[str, Any] = {
                "event_count": len(events),
                "tool_calls": tool_call_count,
                "tool_errors": tool_error_count,
            }
            if project_root is not None:
                metrics["project_root"] = str(project_root)
            # 从 final event 提取运行级指标
            agent_metrics = _extract_agent_metrics(events)
            if agent_metrics:
                metrics.update(agent_metrics)
            memory_metrics = _build_memory_metrics(task, memory_trace)
            if memory_trace:
                metrics["memory_trace"] = [asdict(event) for event in memory_trace]
            if memory_metrics:
                metrics.update(memory_metrics)
            if after_evidence:
                metrics["file_evidence"] = after_evidence
            model_patch = _collect_model_patch(project_root)
            if model_patch:
                metrics["model_patch"] = model_patch
            if validation_results:
                metrics["validation"] = validation_results_to_dict(validation_results)
            return TrialResult(
                task_id=task.id,
                trial_id=trial_id,
                success=success,
                answer=answer,
                trace_path=trace_path,
                graders=graders,
                metrics=metrics,
            )
        finally:
            app.close()


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if hasattr(coro, "close"):
        coro.close()
    raise RuntimeError(
        "EvalRunner.run cannot run inside an active event loop; use arun"
    )


def _collect_file_evidence(
    task: EvalTask,
    project_root: Path | None,
) -> list[dict[str, Any]]:
    specs = _file_evidence_specs(task)
    if not specs or project_root is None:
        return []
    root = project_root.resolve()
    records: list[dict[str, Any]] = []
    for spec in specs:
        rel_path = str(spec.get("path", "")).strip()
        if not rel_path:
            continue
        path = (root / rel_path).resolve()
        record: dict[str, Any] = {
            "path": rel_path,
            "exists": path.is_file(),
        }
        try:
            path.relative_to(root)
        except ValueError:
            record["exists"] = False
            record["error"] = "path outside project root"
            records.append(record)
            continue
        if path.is_file():
            data = path.read_bytes()
            record["sha256"] = hashlib.sha256(data).hexdigest()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = ""
                record["text"] = False
            else:
                record["text"] = True
                record["contains"] = {
                    expected: expected in text
                    for expected in _string_tuple(spec.get("contains", ()))
                }
                record["not_contains"] = {
                    expected: expected not in text
                    for expected in _string_tuple(spec.get("not_contains", ()))
                }
        records.append(record)
    return records


def _grade_file_evidence(
    task: EvalTask,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> tuple[GraderResult, ...]:
    specs = _file_evidence_specs(task)
    if not specs:
        return ()
    before_by_path = {record["path"]: record for record in before}
    after_by_path = {record["path"]: record for record in after}
    graders: list[GraderResult] = []
    for spec in specs:
        rel_path = str(spec.get("path", "")).strip()
        if not rel_path:
            continue
        record = after_by_path.get(rel_path, {"exists": False})
        expected_exists = bool(spec.get("exists", True))
        exists = bool(record.get("exists", False))
        graders.append(
            GraderResult(
                name=f"file_exists:{rel_path}",
                passed=exists is expected_exists,
                details="" if exists is expected_exists else f"exists={exists}",
            )
        )
        for expected, present in record.get("contains", {}).items():
            graders.append(
                GraderResult(
                    name=f"file_contains:{rel_path}:{expected}",
                    passed=bool(present),
                    details="" if present else f"missing {expected!r}",
                )
            )
        for expected, absent in record.get("not_contains", {}).items():
            graders.append(
                GraderResult(
                    name=f"file_not_contains:{rel_path}:{expected}",
                    passed=bool(absent),
                    details="" if absent else f"still contains {expected!r}",
                )
            )
        if "changed" in spec:
            before_sha = before_by_path.get(rel_path, {}).get("sha256")
            after_sha = record.get("sha256")
            changed = before_sha != after_sha
            expected_changed = bool(spec["changed"])
            graders.append(
                GraderResult(
                    name=f"file_changed:{rel_path}",
                    passed=changed is expected_changed,
                    details="" if changed is expected_changed else f"changed={changed}",
                )
            )
    return tuple(graders)


def _file_evidence_specs(task: EvalTask) -> tuple[dict[str, Any], ...]:
    evidence = task.metadata.get("evidence", {})
    if not isinstance(evidence, dict):
        return ()
    files = evidence.get("files", ())
    if not isinstance(files, list | tuple):
        return ()
    return tuple(item for item in files if isinstance(item, dict))


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return ()


def _collect_model_patch(project_root: Path | None) -> str:
    """采集 Git 工作区补丁，供外部 benchmark harness 消费。"""
    if project_root is None:
        return ""
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


_NUMERIC_FIELDS = (
    "llm_calls",
    "estimated_prompt_tokens",
    "model_total_ms",
    "tool_calls",
    "tool_errors",
    "steps",
    "memory_retrieval_count",
    "memory_injected_count",
    "memory_tool_search_count",
    "memory_injected_tokens",
    "memory_retrieval_latency_ms",
)

_MEMORY_RATE_FIELDS = (
    "memory_recall_at_k",
    "memory_mrr",
    "memory_irrelevant_injection_rate",
    "memory_stale_conflict_retrieval_rate",
)

type MemoryEvalConfig = dict[str, Any]


def _extract_agent_metrics(events: list[StructuredAgentEvent]) -> dict[str, Any]:
    for event in events:
        if event.type == "final" and hasattr(event.data, "metrics"):
            agent_m = event.data.metrics
            if not isinstance(agent_m, dict):
                return {}
            metric_names = (
                "llm_calls",
                "estimated_prompt_tokens",
                "model_latencies_ms",
                "tool_latencies_ms",
                "model_total_ms",
                "tool_total_ms",
                "total_observed_ms",
                "steps",
            )
            metrics = {key: agent_m[key] for key in metric_names if key in agent_m}
            metrics["termination_reason"] = event.data.termination_reason.value
            return metrics
    return {}


def _collect_memory_trace(app: XcodeApp) -> tuple[Any, ...]:
    manager = getattr(app, "memory_manager", None)
    if manager is None or not hasattr(manager, "drain_trace_events"):
        return ()
    drained = manager.drain_trace_events()
    return drained if isinstance(drained, tuple) else tuple(drained)


def _build_memory_metrics(task: EvalTask, memory_trace: Sequence[Any]) -> dict[str, Any]:
    if not memory_trace:
        return {}
    retrieved = [
        event for event in memory_trace if getattr(event, "type", "") == "retrieved"
    ]
    injected = [
        event for event in memory_trace if getattr(event, "type", "") == "injected"
    ]
    tool_searched = [
        event for event in memory_trace if getattr(event, "type", "") == "tool_searched"
    ]
    metrics: dict[str, Any] = {
        "memory_retrieval_count": len(retrieved),
        "memory_injected_count": len(injected),
        "memory_tool_search_count": len(tool_searched),
        "memory_injected_tokens": sum(
            int(getattr(event, "token_count", 0) or 0) for event in injected
        ),
    }
    retrieval_latencies = [
        float(getattr(event, "latency_ms", 0.0) or 0.0)
        for event in retrieved
        if getattr(event, "latency_ms", None) is not None
    ]
    if not retrieval_latencies:
        retrieval_latencies = [
            float(getattr(event, "latency_ms", 0.0) or 0.0)
            for event in tool_searched
            if getattr(event, "latency_ms", None) is not None
        ]
    if retrieval_latencies:
        metrics["memory_retrieval_latency_ms"] = round(
            sum(retrieval_latencies) / len(retrieval_latencies),
            3,
        )

    config = task.metadata.get("memory_eval", {})
    if not isinstance(config, dict):
        return metrics

    expected_titles = {
        str(item).strip()
        for item in config.get("expected_titles", ())
        if str(item).strip()
    }
    stale_titles = {
        str(item).strip()
        for item in config.get("stale_or_conflicting_titles", ())
        if str(item).strip()
    }
    retrieved_titles = [str(getattr(event, "title", "") or "") for event in retrieved]
    injected_titles = [str(getattr(event, "title", "") or "") for event in injected]

    if expected_titles:
        retrieved_hits = sum(1 for title in expected_titles if title in retrieved_titles)
        metrics["memory_recall_at_k"] = round(
            retrieved_hits / max(len(expected_titles), 1),
            4,
        )
        reciprocal_rank = 0.0
        for index, title in enumerate(retrieved_titles, start=1):
            if title in expected_titles:
                reciprocal_rank = 1.0 / index
                break
        metrics["memory_mrr"] = round(reciprocal_rank, 4)
        if injected_titles:
            irrelevant = sum(1 for title in injected_titles if title not in expected_titles)
            metrics["memory_irrelevant_injection_rate"] = round(
                irrelevant / len(injected_titles),
                4,
            )
    if stale_titles and retrieved_titles:
        stale_hits = sum(1 for title in retrieved_titles if title in stale_titles)
        metrics["memory_stale_conflict_retrieval_rate"] = round(
            stale_hits / len(retrieved_titles),
            4,
        )
    return metrics


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _build_run_metrics(
    tasks: Sequence[EvalTask], trials: list[TrialResult]
) -> dict[str, Any]:
    passed = sum(1 for t in trials if t.success)
    metrics: dict[str, Any] = {
        "task_count": len(tasks),
        "trial_count": len(trials),
        "passed_trials": passed,
    }
    from collections import defaultdict

    task_trials: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        task_trials[t.task_id].append(t.success)
    k = max(len(v) for v in task_trials.values()) if task_trials else 1
    pass_at_k_rates = [
        _unbiased_pass_at_k(len(values), sum(1 for passed in values if passed), k)
        for values in task_trials.values()
    ]
    pass_at_k_count = sum(1 for v in task_trials.values() if any(v))
    pass_pow_k_count = sum(1 for v in task_trials.values() if all(v))
    metrics["trials_per_task"] = k
    metrics["pass@k"] = f"{pass_at_k_count}/{len(task_trials)}"
    metrics["pass^k"] = f"{pass_pow_k_count}/{len(task_trials)}"
    metrics["pass@k_rate"] = (
        round(sum(pass_at_k_rates) / len(pass_at_k_rates), 4)
        if pass_at_k_rates
        else 0.0
    )
    metrics["pass^k_rate"] = (
        round(pass_pow_k_count / len(task_trials), 4) if task_trials else 0.0
    )
    total_llm = 0
    total_tokens = 0
    total_model_ms = 0.0
    total_tool_calls = 0
    total_tool_errors = 0
    for t in trials:
        total_llm += t.metrics.get("llm_calls", 0)
        total_tokens += t.metrics.get("estimated_prompt_tokens", 0)
        total_model_ms += t.metrics.get("model_total_ms", 0.0)
        total_tool_calls += t.metrics.get("tool_calls", 0)
        total_tool_errors += t.metrics.get("tool_errors", 0)
    if total_llm:
        metrics["total_llm_calls"] = total_llm
    if total_tokens:
        metrics["total_estimated_tokens"] = total_tokens
    if total_model_ms:
        metrics["total_model_ms"] = round(total_model_ms, 1)
    metrics["total_tool_calls"] = total_tool_calls
    metrics["total_tool_errors"] = total_tool_errors
    for field in _NUMERIC_FIELDS:
        values = [t.metrics.get(field, 0) or 0 for t in trials]
        if all(v == 0 for v in values):
            continue
        sv = sorted(values)
        metrics[f"{field}_distribution"] = {
            "min": sv[0],
            "p50": round(_percentile(sv, 50), 1),
            "p95": round(_percentile(sv, 95), 1),
            "p99": round(_percentile(sv, 99), 1),
            "max": sv[-1],
            "mean": round(sum(sv) / len(sv), 1),
        }
    for field in _MEMORY_RATE_FIELDS:
        values = [t.metrics.get(field) for t in trials if t.metrics.get(field) is not None]
        if not values:
            continue
        metrics[f"{field}_mean"] = round(
            sum(float(value) for value in values) / len(values),
            4,
        )
    memory_ablation = _build_memory_ablation_metrics(tasks, trials)
    if memory_ablation:
        metrics.update(memory_ablation)
    all_graders = [grader for trial in trials for grader in trial.graders]
    skipped_graders = [grader for grader in all_graders if grader.skipped]
    evaluated_graders = [grader for grader in all_graders if not grader.skipped]
    if skipped_graders:
        metrics["grader_skipped_count"] = len(skipped_graders)
    if evaluated_graders:
        total_g = len(evaluated_graders)
        passed_g = sum(1 for grader in evaluated_graders if grader.passed)
        metrics["grader_pass_rate"] = round(passed_g / total_g, 4)
        by_name: dict[str, list[bool]] = {}
        for grader in evaluated_graders:
            by_name.setdefault(grader.name, []).append(grader.passed)
        metrics["per_grader_pass_rate"] = {
            name: round(sum(v) / len(v), 4) for name, v in by_name.items()
        }
        task_graders: dict[str, list[bool]] = defaultdict(list)
        for trial in trials:
            for grader in trial.graders:
                if grader.skipped:
                    continue
                task_graders[trial.task_id].append(grader.passed)
        metrics["per_task_grader_rate"] = {
            tid: round(sum(v) / len(v), 4) for tid, v in sorted(task_graders.items())
        }
    return metrics


def _unbiased_pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    """使用 HumanEval 无偏估计量计算 pass@k。"""
    if sample_count <= 0 or correct_count <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct_count, k) / math.comb(sample_count, k)


def _build_memory_ablation_metrics(
    tasks: Sequence[EvalTask],
    trials: Sequence[TrialResult],
) -> dict[str, Any]:
    task_configs = {
        task.id: _memory_eval_config(task)
        for task in tasks
        if _memory_eval_config(task) is not None
    }
    grouped: dict[tuple[str, int], dict[str, TrialResult]] = {}
    for trial in trials:
        config = task_configs.get(trial.task_id)
        if config is None:
            continue
        group = str(config.get("comparison_group", "")).strip()
        mode = str(config.get("mode", "")).strip().lower()
        if not group or mode not in {"on", "off"}:
            continue
        key = (group, _trial_iteration(trial.trial_id))
        grouped.setdefault(key, {})[mode] = trial

    paired = [pair for pair in grouped.values() if {"on", "off"} <= pair.keys()]
    if not paired:
        return {}

    on_successes = [1.0 if pair["on"].success else 0.0 for pair in paired]
    off_successes = [1.0 if pair["off"].success else 0.0 for pair in paired]
    tool_call_deltas = [
        float((pair["on"].metrics.get("tool_calls", 0) or 0))
        - float((pair["off"].metrics.get("tool_calls", 0) or 0))
        for pair in paired
    ]
    negative_migrations = [
        1.0 if (not pair["on"].success and pair["off"].success) else 0.0
        for pair in paired
    ]
    return {
        "memory_ablation_pair_count": len(paired),
        "memory_on_success_rate": round(sum(on_successes) / len(paired), 4),
        "memory_off_success_rate": round(sum(off_successes) / len(paired), 4),
        "memory_success_delta": round(
            (sum(on_successes) - sum(off_successes)) / len(paired),
            4,
        ),
        "memory_tool_call_delta_mean": round(
            sum(tool_call_deltas) / len(tool_call_deltas),
            4,
        ),
        "memory_negative_migration_rate": round(
            sum(negative_migrations) / len(negative_migrations),
            4,
        ),
    }


def _memory_eval_config(task: EvalTask) -> MemoryEvalConfig | None:
    config = task.metadata.get("memory_eval")
    if isinstance(config, dict):
        return config
    return None


def _trial_iteration(trial_id: str) -> int:
    suffix = trial_id.rsplit("-", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 1
