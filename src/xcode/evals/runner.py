from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import datetime, UTC
from pathlib import Path
import hashlib
import uuid
from typing import Any

from xcode.harness.app import XcodeApp
from xcode.harness.agent_runtime import StructuredAgentEvent

from .graders import grade_events, run_llm_judge
from .reporting import write_report_files
from .schema import EvalReport, EvalTask, GraderResult, TrialResult
from .tracing import TraceRecorder

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
        trials: list[TrialResult] = []
        for task in self.tasks:
            for trial_index in range(self.trials_per_task):
                trials.append(await self._run_trial(task, trial_index))
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
        evidence_graders = _grade_file_evidence(task, before_evidence, after_evidence)
        graders = grade_events(task, events, answer, runtime_error) + evidence_graders

        # LLM-as-judge：当 task.llm_judge_criteria 非空时执行
        if task.llm_judge_criteria:
            judge_provider = getattr(app.agent, "provider", None)
            judge_graders = run_llm_judge(
                task, answer, events, judge_provider=judge_provider
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
        # 从 final event 提取运行级指标
        agent_metrics = _extract_agent_metrics(events)
        if agent_metrics:
            metrics.update(agent_metrics)
        if after_evidence:
            metrics["file_evidence"] = after_evidence
        return TrialResult(
            task_id=task.id,
            trial_id=trial_id,
            success=success,
            answer=answer,
            trace_path=trace_path,
            graders=graders,
            metrics=metrics,
        )


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


def _extract_agent_metrics(events: list[StructuredAgentEvent]) -> dict[str, Any]:
    """从 final event 的 metrics 中提取运行级指标。"""
    for event in events:
        if event.type == "final" and hasattr(event.data, "metrics"):
            agent_m = event.data.metrics
            if not isinstance(agent_m, dict):
                return {}
            return {
                k: agent_m[k]
                for k in (
                    "llm_calls",
                    "estimated_prompt_tokens",
                    "model_latencies_ms",
                    "tool_latencies_ms",
                    "model_total_ms",
                    "tool_total_ms",
                    "total_observed_ms",
                )
                if k in agent_m
            }
    return {}


def _build_run_metrics(
    tasks: Sequence[EvalTask], trials: list[TrialResult]
) -> dict[str, Any]:
    """构建 run 级汇总指标：基础计数 + 跨 trial 聚合 + grader 统计。"""
    passed = sum(1 for t in trials if t.success)
    metrics: dict[str, Any] = {
        "task_count": len(tasks),
        "trial_count": len(trials),
        "passed_trials": passed,
    }
    # pass@k: k 次至少一次正确（探索能力上限）
    # pass^k: k 次全部正确（上线回归）
    from collections import defaultdict
    task_trials: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        task_trials[t.task_id].append(t.success)
    k = max(len(v) for v in task_trials.values()) if task_trials else 1
    pass_at_k_count = sum(1 for v in task_trials.values() if any(v))
    pass_pow_k_count = sum(1 for v in task_trials.values() if all(v))
    metrics["trials_per_task"] = k
    metrics["pass@k"] = f"{pass_at_k_count}/{len(task_trials)}"
    metrics["pass^k"] = f"{pass_pow_k_count}/{len(task_trials)}"
    # 跨 trial 聚合延迟和 token
    total_llm = 0
    total_tokens = 0
    total_model_ms = 0.0
    for t in trials:
        total_llm += t.metrics.get("llm_calls", 0)
        total_tokens += t.metrics.get("estimated_prompt_tokens", 0)
        total_model_ms += t.metrics.get("model_total_ms", 0.0)
    if total_llm:
        metrics["total_llm_calls"] = total_llm
    if total_tokens:
        metrics["total_estimated_tokens"] = total_tokens
    if total_model_ms:
        metrics["total_model_ms"] = round(total_model_ms, 1)
    # grader 统计
    all_graders = [g for t in trials for g in t.graders]
    if all_graders:
        total_g = len(all_graders)
        passed_g = sum(1 for g in all_graders if g.passed)
        metrics["grader_pass_rate"] = round(passed_g / total_g, 4)
        # 按 grader name 分组统计
        by_name: dict[str, list[bool]] = {}
        for g in all_graders:
            by_name.setdefault(g.name, []).append(g.passed)
        metrics["per_grader_pass_rate"] = {
            name: round(sum(v) / len(v), 4) for name, v in by_name.items()
        }
    return metrics
