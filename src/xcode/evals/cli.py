from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator, Callable, Mapping
import json
from pathlib import Path
import tempfile
from typing import Any

from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.ai.events import (
    FinalMessage,
    Message,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.ai.providers.protocol import ModelProvider
from xcode.harness.app import XcodeApp, build_app as build_real_app
from xcode.harness.config import AgentConfig, discover_runtime_config
from xcode.harness.memory import MemoryManager
from xcode.harness.observability import HITLResult
from xcode.harness.skills import ToolSpec

from .adapters.registry import BENCHMARK_ADAPTERS, INTEGRATED_BENCHMARKS
from .benchmarks import load_benchmark
from .runner import EvalRunner
from .sandbox import trial_project_root
from .schema import EvalReport, EvalTask, EvalTaskSchemaError, parse_eval_task
from .tasks import SUITE_DESCRIPTIONS, SUITE_SPECS, SUITES


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir or Path.cwd() / ".local" / "eval_runs"
    if args.list_suites:
        _print_suite_list()
        return 0
    if args.list_benchmarks:
        _print_benchmark_list()
        return 0
    if args.show_suite:
        return _print_suite_detail(args.show_suite)
    tasks: tuple[EvalTask, ...]
    task_source = "smoke"
    if args.suite:
        suite = SUITES.get(args.suite)
        if suite is None:
            available = ", ".join(sorted(SUITES))
            print(f"Unknown suite: {args.suite}. Available: {available}")
            return 1
        tasks = suite
        task_source = f"suite:{args.suite}"
    elif args.benchmark:
        if args.benchmark_path is None:
            print("--benchmark-path is required when --benchmark is set")
            return 1
        try:
            tasks = load_benchmark(
                args.benchmark,
                args.benchmark_path,
                fixture_root=output_dir / "benchmark_fixtures",
                limit=args.limit,
            )
        except ValueError as exc:
            print(str(exc))
            return 1
        task_source = f"benchmark:{args.benchmark}"
    elif args.tasks:
        try:
            tasks = _load_tasks(args.tasks)
        except (OSError, json.JSONDecodeError, EvalTaskSchemaError) as exc:
            print(str(exc))
            return 1
        task_source = f"tasks:{args.tasks}"
    else:
        tasks = SUITES.get("smoke", ())
    runner = EvalRunner(
        tasks=tasks,
        app_factory=_real_app_factory(
            args.project_root,
            output_dir,
            allow_project_mutation=args.allow_project_mutation,
        )
        if args.real
        else _offline_app_factory,
        output_dir=output_dir,
        trials_per_task=args.trials,
        suite_name=args.suite,
        task_source=task_source,
    )
    report = runner.run()
    print(f"Eval run: {report.run_id}")
    print(f"Status: {'PASS' if report.success else 'FAIL'}")
    _print_enhanced_summary(report)
    if args.baseline:
        comparison = _print_baseline_diff(report, args.baseline)
        _write_baseline_diff_artifact(report.output_dir, comparison)
        gate_failures = _evaluate_baseline_gates(comparison, args)
        if gate_failures:
            print("\nCI Gate:")
            for failure in gate_failures:
                print(f"  - {failure}")
            return 1
    _print_failed_trials(report)
    print(f"Manifest:    {report.output_dir / 'run_manifest.json'}")
    print(f"Report JSON: {report.output_dir / 'report.json'}")
    print(f"Report HTML: {report.output_dir / 'report.html'}")
    print(f"Report CSV:  {report.output_dir / 'report.csv'}")
    return 0 if report.success else 1


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


def _fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _print_enhanced_summary(report: EvalReport) -> None:
    m = report.metrics
    _print_task_table(report)
    pass_at_k = m.get("pass@k")
    pass_k_rate = m.get("pass@k_rate")
    pass_pow_k = m.get("pass^k")
    pass_pow_rate = m.get("pass^k_rate")
    if pass_at_k and pass_pow_k:
        rate_str = f" ({pass_k_rate * 100:.0f}%)" if pass_k_rate is not None else ""
        pow_str = f" ({pass_pow_rate * 100:.0f}%)" if pass_pow_rate is not None else ""
        print(f"pass@k: {pass_at_k}{rate_str}   pass^k: {pass_pow_k}{pow_str}")
    grader_rate = m.get("grader_pass_rate")
    if grader_rate is not None:
        evaluated = [
            grader
            for trial in report.trials
            for grader in trial.graders
            if not grader.skipped
        ]
        passed_g = sum(1 for grader in evaluated if grader.passed)
        print(f"Graders: {passed_g}/{len(evaluated)} ({grader_rate * 100:.1f}%)")
        _print_grader_bars(m.get("per_grader_pass_rate", {}))
    skipped_count = m.get("grader_skipped_count", 0)
    if skipped_count:
        print(f"Skipped graders: {skipped_count}")
    parts = []
    total_llm = m.get("total_llm_calls", 0)
    if total_llm:
        parts.append(f"{total_llm} LLM calls")
    total_tokens = m.get("total_estimated_tokens", 0)
    if total_tokens:
        parts.append(f"~{total_tokens:,} tokens")
    total_ms = m.get("total_model_ms", 0.0)
    if total_ms:
        parts.append(f"{total_ms / 1000:.1f}s model time")
    if parts:
        print(f"Metrics: {', '.join(parts)}")
    if m.get("memory_ablation_pair_count"):
        print(
            "Memory on/off: "
            f"pairs={m['memory_ablation_pair_count']}  "
            f"on={m.get('memory_on_success_rate', 0.0) * 100:.1f}%  "
            f"off={m.get('memory_off_success_rate', 0.0) * 100:.1f}%  "
            f"delta={m.get('memory_success_delta', 0.0) * 100:.1f}%  "
            f"negative_migration={m.get('memory_negative_migration_rate', 0.0) * 100:.1f}%"
        )
    _print_distribution(m)


def _print_task_table(report: EvalReport) -> None:
    trials = report.trials
    if not trials:
        return
    task_map: dict[str, list] = OrderedDict()
    task_specs = {task.id: task for task in report.tasks}
    for trial in trials:
        task_map.setdefault(trial.task_id, []).append(trial)
    print(f"\nTasks ({len(task_map)} total):")
    id_w = min(max(max(len(t.task_id) for t in trials) + 2, 12), 30)
    for task_id, task_trials in task_map.items():
        best = next((trial for trial in task_trials if trial.success), task_trials[0])
        evaluated = [grader for grader in best.graders if not grader.skipped]
        g_pass = sum(1 for grader in evaluated if grader.passed)
        g_total = len(evaluated)
        status = "PASS" if best.success else "FAIL"
        calls = best.metrics.get("tool_calls", "")
        tokens = best.metrics.get("estimated_prompt_tokens", "")
        if isinstance(tokens, int):
            tokens = f"{tokens:,}"
        lat = best.metrics.get("model_total_ms", 0)
        lat_str = _fmt_ms(float(lat)) if isinstance(lat, int | float) and lat else ""
        mark = "+" if best.success else "x"
        task = task_specs.get(task_id)
        capability = task.capability if task is not None else "general"
        print(
            f"  {mark} {task_id:<{id_w}} {status:>4}  "
            f"graders {g_pass}/{g_total}  "
            f"calls {str(calls):>3}  "
            f"tokens {str(tokens):>7}  "
            f"{lat_str:<7}  "
            f"{capability}"
        )


def _print_grader_bars(per_grader: dict[str, float]) -> None:
    if not per_grader:
        return
    print()
    width = 20
    name_w = min(max(len(k) for k in per_grader) + 2, 50)
    for name, rate in sorted(per_grader.items()):
        fill = int(rate * width)
        bar = "#" * fill + "." * (width - fill)
        print(f"  {name:<{name_w}} [{bar}] {rate * 100:.0f}%")


def _print_distribution(m: dict[str, Any]) -> None:
    rows = []
    for field in _NUMERIC_FIELDS:
        dist = m.get(f"{field}_distribution")
        if not dist:
            continue
        rows.append(
            f"  {field:<22} min={dist['min']}  "
            f"p50={dist['p50']}  p95={dist['p95']}  "
            f"p99={dist['p99']}  max={dist['max']}  mean={dist['mean']}"
        )
    if rows:
        print("\nDistribution:")
        for row in rows:
            print(row)


def _print_failed_trials(report: EvalReport) -> None:
    failed_trials = [trial for trial in report.trials if not trial.success]
    if not failed_trials:
        return
    print("\nFailures:")
    for trial in failed_trials:
        print(f"  - {trial.trial_id}")
        project_root = trial.metrics.get("project_root")
        if project_root:
            print(f"    project_root: {project_root}")
        print(f"    trace: {trial.trace_path}")
        for grader in trial.graders:
            if grader.passed:
                continue
            detail = f": {grader.details}" if grader.details else ""
            print(f"    grader: {grader.name}{detail}")
        validation = trial.metrics.get("validation", ())
        if isinstance(validation, list):
            for item in validation:
                if not isinstance(item, Mapping):
                    continue
                command = item.get("command", "")
                returncode = item.get("returncode", "")
                print(f"    validation: exit={returncode} command={command}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Xcode eval tasks.")
    parser.add_argument(
        "--suite",
        type=str,
        help=f"Run a named task suite: {', '.join(sorted(SUITES))}",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List built-in eval suites and exit.",
    )
    parser.add_argument(
        "--show-suite",
        type=str,
        help="Show tasks in a built-in eval suite and exit.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List external benchmark adapter targets and exit.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        help="JSON or JSONL EvalTask file. If omitted, runs smoke suite.",
    )
    parser.add_argument(
        "--benchmark",
        choices=INTEGRATED_BENCHMARKS,
        help="Load tasks from an integrated benchmark source.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        help="Local path or URL for --benchmark.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run tasks against build_real_app() instead of the offline fake provider.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used by --real.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for trace files, report.json, and report.html.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline report.json or run directory for regression comparison.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Fail the run if candidate regresses vs baseline on any comparable task.",
    )
    parser.add_argument(
        "--max-p95-model-ms-growth",
        type=float,
        default=None,
        help="Fail if candidate p95 model latency grows beyond this many ms vs baseline.",
    )
    parser.add_argument(
        "--max-avg-token-growth",
        type=float,
        default=None,
        help="Fail if candidate average token usage grows beyond this amount vs baseline.",
    )
    parser.add_argument(
        "--max-avg-tool-calls-growth",
        type=float,
        default=None,
        help="Fail if candidate average tool calls grow beyond this amount vs baseline.",
    )
    parser.add_argument(
        "--require-grader-pass",
        action="append",
        default=[],
        help="Require the named grader pass rate to remain 100%% in the candidate run.",
    )
    parser.add_argument(
        "--allow-project-mutation",
        action="store_true",
        help=(
            "Allow --real eval tasks without fixture_dir to modify --project-root. "
            "By default real eval writes must run in sandboxes."
        ),
    )
    parser.add_argument("--trials", type=int, default=1, help="Trials per task.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max tasks to run from the benchmark (e.g. --limit 1 for a quick smoke).",
    )
    return parser.parse_args(argv)


def _print_suite_list() -> None:
    print("Built-in eval suites:")
    for name in sorted(SUITE_SPECS):
        spec = SUITE_SPECS[name]
        print(
            f"  {name:<16} {len(spec.tasks):>2} tasks  "
            f"{spec.kind:<10} {spec.run_mode:<8} {spec.description}"
        )


def _print_suite_detail(name: str) -> int:
    spec = SUITE_SPECS.get(name)
    if spec is None:
        available = ", ".join(sorted(SUITES))
        print(f"Unknown suite: {name}. Available: {available}")
        return 1
    print(f"Suite: {name}")
    print(f"Description: {spec.description}")
    print(f"Kind: {spec.kind}")
    print(f"Run mode: {spec.run_mode}")
    print(f"Tasks ({len(spec.tasks)}):")
    for task in spec.tasks:
        print(f"  - {task.id}")
        print(f"    capability: {task.capability}")
        print(f"    version: {task.version}")
        print(f"    owner: {task.owner}")
        print(f"    difficulty: {task.difficulty}")
        print(f"    run_mode: {task.run_mode}")
        if task.expected_duration_seconds is not None:
            print(f"    expected_duration_seconds: {task.expected_duration_seconds}")
        fixture = task.metadata.fixture_dir
        if fixture:
            print(f"    fixture: {fixture}")
        validation = task.metadata.validation
        if validation is not None and validation.commands:
            print(f"    validation_commands: {len(validation.commands)}")
    return 0


def _print_benchmark_list() -> None:
    print("External benchmark adapter targets:")
    for name in sorted(BENCHMARK_ADAPTERS):
        spec = BENCHMARK_ADAPTERS[name]
        print(f"  {spec.name:<18} {spec.display_name}")
        print(f"    status: {spec.status}")
        print(f"    purpose: {spec.purpose}")
        print(f"    harness: {spec.harness}")
        print(f"    xcode_role: {spec.xcode_role}")
        print(f"    upstream: {spec.upstream_url}")


def _load_tasks(path: Path) -> tuple[EvalTask, ...]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        raw_items = raw if isinstance(raw, list) else [raw]
    tasks = tuple(_task_from_dict(item, index=index) for index, item in enumerate(raw_items))
    _ensure_unique_task_ids(tasks, source=str(path))
    return tasks


def _task_from_dict(item: dict[str, Any], *, index: int = 0) -> EvalTask:
    return parse_eval_task(item, path=f"tasks[{index}]")


def _ensure_unique_task_ids(tasks: tuple[EvalTask, ...], *, source: str) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise EvalTaskSchemaError(f"{source}: duplicate task id {task.id!r}")
        seen.add(task.id)


def _real_app_factory(
    project_root: Path,
    output_dir: Path,
    *,
    allow_project_mutation: bool = False,
) -> Callable[[EvalTask, int], XcodeApp]:
    base_root = project_root.resolve()
    runtime_config = discover_runtime_config(base_root)
    pkg_root = Path(__file__).resolve().parent.parent
    env_files = (
        pkg_root / ".env",
        base_root / ".env",
        base_root / "xcode" / ".env",
    )

    def build(task: EvalTask, trial_index: int) -> XcodeApp:
        effective_root = _trial_project_root(
            task,
            trial_index,
            base_root=base_root,
            output_dir=output_dir,
            allow_project_mutation=allow_project_mutation,
        )
        app = build_real_app(
            project_root=effective_root,
            runtime_config=runtime_config,
            env_files=env_files,
        )
        app.agent.approval_callback = _auto_approve
        return app

    return build


def _auto_approve(tool, input):
    """eval 专用：自动批准所有需要审批的工具调用。"""
    return HITLResult(decision="allow", scope="session")


def _trial_project_root(
    task: EvalTask,
    trial_index: int,
    base_root: Path,
    output_dir: Path,
    allow_project_mutation: bool = False,
) -> Path:
    return trial_project_root(
        task,
        trial_index,
        base_root=base_root,
        output_dir=output_dir,
        allow_project_mutation=allow_project_mutation,
    )


def _offline_app_factory(task: EvalTask, _trial_index: int) -> XcodeApp:
    memory_config = task.metadata.get("memory_eval")
    if isinstance(memory_config, dict):
        return _offline_memory_app_factory(task, memory_config)
    fault_config = task.metadata.get("fault_injection")
    if isinstance(fault_config, dict):
        return _offline_fault_injection_app_factory(task, fault_config)
    scripted = _offline_scripted_tool_app(task)
    if scripted is not None:
        return scripted
    answer = " ".join(task.expected_answer_contains) or "offline ok"
    return XcodeApp(
        agent=StructuredAgent(
            provider=_StaticProvider(
                [[TextDelta(chunk=answer), FinalMessage(content=answer, stop_reason="end_turn")]]
            ),
            registry=(),
            config=AgentConfig(max_steps=3),
        ),
        registry=(),
    )


def _offline_scripted_tool_app(task: EvalTask) -> XcodeApp | None:
    policy = task.metadata.get("tool_policy")
    policy_dict = policy if isinstance(policy, dict) else {}
    tool_sequence = list(task.expected_tool_calls)
    if not tool_sequence:
        tool_sequence = list(_string_tuple(policy_dict.get("ordered_tools", ())))
    if not tool_sequence:
        return None
    argument_checks = _dict_tuple(policy_dict.get("argument_contains", ()))
    result_checks = _dict_tuple(policy_dict.get("result_contains", ()))
    adopted_checks = _dict_tuple(policy_dict.get("answer_contains_from_tool", ()))
    tool_inputs = _build_offline_tool_inputs(tool_sequence, argument_checks, task.prompt)
    tool_outputs = _build_offline_tool_outputs(tool_sequence, result_checks)
    turns: list[list[ProviderEvent]] = []
    handlers: dict[str, list[object]] = {
        tool_name: list(outputs) for tool_name, outputs in tool_outputs.items()
    }
    registry: list[ToolSpec] = []
    seen_tools: set[str] = set()
    for index, tool_name in enumerate(tool_sequence, start=1):
        turns.append(
            [
                ToolCallEvent(
                    [
                        ToolCall(
                            id=f"{task.id}-call-{index}",
                            name=tool_name,
                            input=tool_inputs[index - 1],
                        )
                    ]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
        )
        if tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        registry.append(
            ToolSpec(
                name=tool_name,
                description="Offline eval scripted tool.",
                input_hint="text",
                handler=_offline_tool_handler(tool_name, handlers),
                read_only=True,
                concurrency_safe=True,
                schema={
                    "type": "object",
                    "additionalProperties": True,
                },
            )
        )
    answer = _offline_answer(task, adopted_checks)
    turns.append(
        [
            TextDelta(chunk=answer),
            FinalMessage(content=answer, stop_reason="end_turn"),
        ]
    )
    return XcodeApp(
        agent=StructuredAgent(
            provider=_StaticProvider(turns),
            registry=tuple(registry),
            config=AgentConfig(max_steps=max(3, len(turns))),
        ),
        registry=tuple(registry),
    )


def _offline_fault_injection_app_factory(
    task: EvalTask,
    config: dict[str, Any],
) -> XcodeApp:
    scenario = str(config.get("scenario", "")).strip()
    if scenario == "command_failure_retry":
        return _offline_fault_scripted_app(
            task,
            tool_sequence=("run_tests", "run_tests"),
            tool_inputs=(
                {"command": "pytest -q"},
                {"command": "python -m pytest -q"},
            ),
            tool_outputs={
                "run_tests": [
                    RuntimeError("pytest: command not found"),
                    "2 passed in 0.01s",
                ],
            },
            answer="pytest rerun passed: 2 passed",
        )
    if scenario == "wrong_path_recovery":
        return _offline_fault_scripted_app(
            task,
            tool_sequence=("read_file", "grep_search", "read_file"),
            tool_inputs=(
                {"path": "src/xcode/evals/runner_missing.py"},
                {"query": "EvalRunner", "path": "src/xcode/evals"},
                {"path": "src/xcode/evals/runner.py"},
            ),
            tool_outputs={
                "read_file": [
                    FileNotFoundError("src/xcode/evals/runner_missing.py not found"),
                    "class EvalRunner:\n    pass\n",
                ],
                "grep_search": ["src/xcode/evals/runner.py: class EvalRunner"],
            },
            answer="Recovered with grep_search and opened runner.py for EvalRunner",
        )
    if scenario == "provider_abort_degrade":
        answer = "interrupted by provider; retry safely"
        return XcodeApp(
            agent=StructuredAgent(
                provider=_StaticProvider(
                    [[TextDelta(chunk=answer), FinalMessage(content=answer, stop_reason="aborted")]]
                ),
                registry=(),
                config=AgentConfig(max_steps=2),
            ),
            registry=(),
        )
    return _offline_scripted_tool_app(task) or XcodeApp(
        agent=StructuredAgent(
            provider=_StaticProvider(
                [[TextDelta(chunk="offline fault"), FinalMessage(content="offline fault", stop_reason="end_turn")]]
            ),
            registry=(),
            config=AgentConfig(max_steps=2),
        ),
        registry=(),
    )


def _offline_fault_scripted_app(
    task: EvalTask,
    *,
    tool_sequence: tuple[str, ...],
    tool_inputs: tuple[dict[str, Any], ...],
    tool_outputs: dict[str, list[object]],
    answer: str,
) -> XcodeApp:
    turns: list[list[ProviderEvent]] = []
    registry: list[ToolSpec] = []
    seen_tools: set[str] = set()
    for index, tool_name in enumerate(tool_sequence, start=1):
        turns.append(
            [
                ToolCallEvent(
                    [
                        ToolCall(
                            id=f"{task.id}-call-{index}",
                            name=tool_name,
                            input=tool_inputs[index - 1],
                        )
                    ]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
        )
        if tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        registry.append(
            ToolSpec(
                name=tool_name,
                description="Offline eval fault-injection tool.",
                input_hint="text",
                handler=_offline_tool_handler(tool_name, tool_outputs),
                read_only=True,
                concurrency_safe=True,
                schema={"type": "object", "additionalProperties": True},
            )
        )
    turns.append(
        [TextDelta(chunk=answer), FinalMessage(content=answer, stop_reason="end_turn")]
    )
    return XcodeApp(
        agent=StructuredAgent(
            provider=_StaticProvider(turns),
            registry=tuple(registry),
            config=AgentConfig(max_steps=max(4, len(turns))),
        ),
        registry=tuple(registry),
    )


def _build_offline_tool_inputs(
    tool_sequence: list[str],
    argument_checks: tuple[dict[str, Any], ...],
    prompt: str,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    remaining = list(argument_checks)
    for tool_name in tool_sequence:
        matched: dict[str, Any] | None = None
        for check in remaining:
            if str(check.get("tool", "")).strip() == tool_name and isinstance(
                check.get("arguments"),
                dict,
            ):
                matched = dict(check["arguments"])
                remaining.remove(check)
                break
        inputs.append(matched or {"input": prompt})
    return inputs


def _build_offline_tool_outputs(
    tool_sequence: list[str],
    result_checks: tuple[dict[str, Any], ...],
) -> dict[str, list[object]]:
    outputs: dict[str, list[object]] = {}
    remaining = list(result_checks)
    for tool_name in tool_sequence:
        text = tool_name
        matched_index: int | None = None
        for index, check in enumerate(remaining):
            if str(check.get("tool", "")).strip() != tool_name:
                continue
            substrings = _string_tuple(check.get("substrings", ()))
            if substrings:
                text = " ".join(substrings)
            matched_index = index
            break
        if matched_index is not None:
            remaining.pop(matched_index)
        outputs.setdefault(tool_name, []).append(text)
    return outputs


def _offline_tool_handler(
    tool_name: str,
    scripted_outputs: dict[str, list[object]],
) -> Callable[[dict[str, Any]], str]:
    def handler(_value: dict[str, Any]) -> str:
        queue = scripted_outputs.setdefault(tool_name, [])
        next_value = queue.pop(0) if queue else tool_name
        if isinstance(next_value, BaseException):
            raise next_value
        return str(next_value)

    return handler


def _offline_answer(
    task: EvalTask,
    adopted_checks: tuple[dict[str, Any], ...],
) -> str:
    if task.expected_answer_contains:
        return " ".join(task.expected_answer_contains)
    snippets: list[str] = []
    for check in adopted_checks:
        snippets.extend(_string_tuple(check.get("substrings", ())))
    return " ".join(snippets) or "finished"


def _dict_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
    return tuple(result)


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, list | tuple):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _offline_memory_app_factory(
    task: EvalTask,
    memory_config: dict[str, Any],
) -> XcodeApp:
    temp_dir = tempfile.TemporaryDirectory(prefix="xcode-eval-memory-")
    project_root = Path(temp_dir.name)
    manager = MemoryManager(
        project_root,
        user_memory_file=project_root / "home" / ".xcode" / "memory" / "MEMORY.md",
    )
    for block in memory_config.get("offline_memory_blocks", ()):
        text = str(block).strip()
        if text:
            manager.add_memory_block(text, source="offline-eval")
    manager.drain_trace_events()
    answer = " ".join(task.expected_answer_contains) or "offline ok"
    provider = _StaticProvider(
        [[TextDelta(chunk=answer), FinalMessage(content=answer, stop_reason="end_turn")]]
    )
    runtime = AgentRuntimeConfig(
        project_root=project_root,
        runtime_context_provider=build_runtime_context_provider(
            project_root,
            (),
            memory_manager=manager,
        )
        if str(memory_config.get("mode", "")).strip().lower() == "on"
        else None,
    )
    return XcodeApp(
        agent=StructuredAgent(
            provider=provider,
            registry=(),
            config=AgentConfig(max_steps=3),
            runtime=runtime,
        ),
        registry=(),
        memory_manager=manager,
        _closers=(temp_dir.cleanup,),
    )


def _print_baseline_diff(report: EvalReport, baseline_path: Path) -> dict[str, Any]:
    baseline_data = _load_report_file(baseline_path)
    comparison = _compare_report_to_baseline(report, baseline_data)
    print("\nBaseline:")
    print(f"  source: {baseline_data.get('run_id', baseline_path)}")
    print(
        f"  regression={len(comparison['regression'])}  "
        f"improvement={len(comparison['improvement'])}  "
        f"unchanged={len(comparison['unchanged'])}  "
        f"incomparable={len(comparison['incomparable'])}"
    )
    for key in ("regression", "improvement", "unchanged", "incomparable"):
        values = comparison[key]
        if values:
            print(f"  {key}: {', '.join(values)}")
    slices = comparison["capability_slices"]
    if slices:
        print("  capability_slices:")
        for capability, row in sorted(slices.items()):
            print(
                f"    {capability}: baseline={row['baseline']:.1%} "
                f"candidate={row['candidate']:.1%}"
            )
    per_grader = comparison["per_grader_delta"]
    if per_grader:
        print("  grader_delta:")
        for name, delta in sorted(per_grader.items()):
            print(f"    {name}: {delta:+.1%}")
    summary = comparison["summary_delta"]
    if summary:
        print("  summary_delta:")
        for key, value in summary.items():
            print(f"    {key}: {value}")
    failures = comparison["failure_category_delta"]
    if failures:
        print("  failure_category_delta:")
        for name, value in sorted(failures.items()):
            print(f"    {name}: {value:+.1%}")
    return comparison


def _load_report_file(path: Path) -> dict[str, Any]:
    report_path = path / "report.json" if path.is_dir() else path
    return json.loads(report_path.read_text(encoding="utf-8"))


def _compare_report_to_baseline(
    report: EvalReport,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    candidate_trials = _best_trial_map_from_report(report)
    baseline_trials = _best_trial_map_from_dict(baseline)
    task_ids = sorted(set(candidate_trials) | set(baseline_trials))
    comparison: dict[str, Any] = {
        "regression": [],
        "improvement": [],
        "unchanged": [],
        "incomparable": [],
        "capability_slices": {},
        "per_grader_delta": {},
        "summary_delta": {},
        "failure_category_delta": {},
        "candidate_grader_rates": report.metrics.get("per_grader_pass_rate", {}),
        "task_details": {},
    }
    for task_id in task_ids:
        candidate_trial = candidate_trials.get(task_id)
        baseline_trial = baseline_trials.get(task_id)
        if candidate_trial is None or baseline_trial is None:
            comparison["incomparable"].append(task_id)
            comparison["task_details"][task_id] = _task_diff_details(
                task_id,
                candidate_trial,
                baseline_trial,
            )
            continue
        if baseline_trial["success"] and not candidate_trial["success"]:
            comparison["regression"].append(task_id)
        elif not baseline_trial["success"] and candidate_trial["success"]:
            comparison["improvement"].append(task_id)
        else:
            comparison["unchanged"].append(task_id)
        comparison["task_details"][task_id] = _task_diff_details(
            task_id,
            candidate_trial,
            baseline_trial,
        )
    comparison["capability_slices"] = _compare_capability_slices(report, baseline)
    comparison["per_grader_delta"] = _compare_grader_rates(report, baseline)
    comparison["summary_delta"] = _compare_summary_metrics(report, baseline)
    comparison["failure_category_delta"] = _compare_failure_categories(report, baseline)
    return comparison


def _best_trial_map_from_report(report: EvalReport) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for trial in report.trials:
        grouped[trial.task_id].append(trial)
    result: dict[str, dict[str, Any]] = {}
    for task_id, trials in grouped.items():
        best = next((trial for trial in trials if trial.success), trials[0])
        result[task_id] = {
            "success": best.success,
            "metrics": best.metrics,
            "graders": tuple(best.graders),
            "trace_path": str(best.trace_path),
            "answer": best.answer,
        }
    return result


def _best_trial_map_from_dict(report_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in report_data.get("trials", ()):
        if isinstance(trial, Mapping) and "task_id" in trial:
            grouped[str(trial["task_id"])].append(dict(trial))
    result: dict[str, dict[str, Any]] = {}
    for task_id, trials in grouped.items():
        best = next((trial for trial in trials if trial.get("success")), trials[0])
        result[task_id] = best
    return result


def _task_diff_details(
    task_id: str,
    candidate_trial: dict[str, Any] | None,
    baseline_trial: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "candidate": _normalize_trial_detail(candidate_trial),
        "baseline": _normalize_trial_detail(baseline_trial),
        "metric_delta": _task_metric_delta(candidate_trial, baseline_trial),
    }


def _normalize_trial_detail(trial: dict[str, Any] | None) -> dict[str, Any] | None:
    if trial is None:
        return None
    metrics = trial.get("metrics", {})
    graders = trial.get("graders", ())
    return {
        "success": trial.get("success"),
        "trace_path": trial.get("trace_path"),
        "model_patch": metrics.get("model_patch", ""),
        "validation": metrics.get("validation", []),
        "failing_graders": [
            {
                "name": _grader_attr(grader, "name"),
                "details": _grader_attr(grader, "details", ""),
                "failure_category": _grader_attr(grader, "failure_category"),
                "evidence": _grader_attr(grader, "evidence"),
            }
            for grader in graders
            if not bool(_grader_attr(grader, "passed", False))
        ],
        "failure_categories": sorted(
            {
                str(_grader_attr(grader, "failure_category"))
                for grader in graders
                if _grader_attr(grader, "failure_category")
            }
        ),
        "metrics": {
            "tool_calls": metrics.get("tool_calls", "unavailable"),
            "estimated_prompt_tokens": metrics.get("estimated_prompt_tokens", "unavailable"),
            "model_total_ms": metrics.get("model_total_ms", "unavailable"),
            "wall_clock_ms": metrics.get("wall_clock_ms", "unavailable"),
            "termination_reason": metrics.get("termination_reason", "unavailable"),
        },
    }


def _task_metric_delta(
    candidate_trial: dict[str, Any] | None,
    baseline_trial: dict[str, Any] | None,
) -> dict[str, Any]:
    if candidate_trial is None or baseline_trial is None:
        return {}
    fields = (
        "tool_calls",
        "estimated_prompt_tokens",
        "model_total_ms",
        "wall_clock_ms",
    )
    candidate_metrics = candidate_trial.get("metrics", {})
    baseline_metrics = baseline_trial.get("metrics", {})
    result: dict[str, Any] = {}
    for field in fields:
        candidate_value = candidate_metrics.get(field)
        baseline_value = baseline_metrics.get(field)
        if not isinstance(candidate_value, int | float) or not isinstance(
            baseline_value, int | float
        ):
            result[field] = "unavailable"
            continue
        result[field] = round(float(candidate_value) - float(baseline_value), 4)
    return result


def _grader_attr(grader: object, key: str, default: Any = None) -> Any:
    if isinstance(grader, Mapping):
        return grader.get(key, default)
    return getattr(grader, key, default)


def _compare_capability_slices(
    report: EvalReport,
    baseline: dict[str, Any],
) -> dict[str, dict[str, float]]:
    baseline_tasks = {
        str(task.get("id")): str(task.get("capability", "general"))
        for task in baseline.get("tasks", ())
        if isinstance(task, Mapping)
    }
    baseline_trials = _best_trial_map_from_dict(baseline)
    candidate_trials = _best_trial_map_from_report(report)
    rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"baseline": [], "candidate": []}
    )
    for task in report.tasks:
        baseline_trial = baseline_trials.get(task.id)
        candidate_trial = candidate_trials.get(task.id)
        capability = task.capability
        if baseline_trial is None or candidate_trial is None:
            continue
        rows[capability]["candidate"].append(1.0 if candidate_trial["success"] else 0.0)
        rows[capability]["baseline"].append(1.0 if baseline_trial.get("success") else 0.0)
    for task_id, capability in baseline_tasks.items():
        if any(task.id == task_id for task in report.tasks):
            continue
        baseline_trial = baseline_trials.get(task_id)
        if baseline_trial is None:
            continue
        rows[capability]["baseline"].append(1.0 if baseline_trial.get("success") else 0.0)
    result: dict[str, dict[str, float]] = {}
    for capability, values in rows.items():
        if not values["baseline"] or not values["candidate"]:
            continue
        result[capability] = {
            "baseline": sum(values["baseline"]) / len(values["baseline"]),
            "candidate": sum(values["candidate"]) / len(values["candidate"]),
        }
    return result


def _compare_grader_rates(
    report: EvalReport,
    baseline: dict[str, Any],
) -> dict[str, float]:
    baseline_rates = baseline.get("metrics", {}).get("per_grader_pass_rate", {})
    candidate_rates = report.metrics.get("per_grader_pass_rate", {})
    if not isinstance(baseline_rates, Mapping) or not isinstance(candidate_rates, Mapping):
        return {}
    deltas: dict[str, float] = {}
    for name in sorted(set(baseline_rates) & set(candidate_rates)):
        deltas[str(name)] = float(candidate_rates[name]) - float(baseline_rates[name])
    return deltas


def _compare_summary_metrics(
    report: EvalReport,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = baseline.get("metrics", {})
    if not isinstance(baseline_metrics, Mapping):
        return {}
    candidate_metrics = report.metrics
    fields = {
        "avg_tokens": (
            _average_from_total_and_count(candidate_metrics, "total_estimated_tokens"),
            _average_from_total_and_count(baseline_metrics, "total_estimated_tokens"),
        ),
        "avg_tool_calls": (
            _average_from_total_and_count(candidate_metrics, "total_tool_calls"),
            _average_from_total_and_count(baseline_metrics, "total_tool_calls"),
        ),
        "avg_model_ms": (
            _average_from_total_and_count(candidate_metrics, "total_model_ms"),
            _average_from_total_and_count(baseline_metrics, "total_model_ms"),
        ),
        "p95_model_ms": (
            _distribution_value(candidate_metrics, "model_total_ms_distribution", "p95"),
            _distribution_value(baseline_metrics, "model_total_ms_distribution", "p95"),
        ),
        "avg_token_cost": ("unavailable", "unavailable"),
    }
    result: dict[str, Any] = {}
    for name, (candidate, base) in fields.items():
        if candidate == "unavailable" or base == "unavailable":
            result[name] = "unavailable"
            continue
        result[name] = round(float(candidate) - float(base), 4)
    return result


def _compare_failure_categories(
    report: EvalReport,
    baseline: dict[str, Any],
) -> dict[str, float]:
    candidate_rates = report.metrics.get("failure_category_pass_rate", {})
    baseline_rates = baseline.get("metrics", {}).get("failure_category_pass_rate", {})
    if not isinstance(candidate_rates, Mapping) or not isinstance(baseline_rates, Mapping):
        return {}
    result: dict[str, float] = {}
    for name in sorted(set(candidate_rates) & set(baseline_rates)):
        result[str(name)] = float(candidate_rates[name]) - float(baseline_rates[name])
    return result


def _average_from_total_and_count(metrics: Mapping[str, Any], total_key: str) -> Any:
    total = metrics.get(total_key)
    count = metrics.get("trial_count")
    if not isinstance(total, int | float) or not isinstance(count, int) or count <= 0:
        return "unavailable"
    return float(total) / count


def _distribution_value(
    metrics: Mapping[str, Any],
    key: str,
    field: str,
) -> Any:
    value = metrics.get(key)
    if not isinstance(value, Mapping):
        return "unavailable"
    selected = value.get(field)
    if not isinstance(selected, int | float):
        return "unavailable"
    return float(selected)


def _evaluate_baseline_gates(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    regressions = comparison.get("regression", [])
    if args.fail_on_regression and regressions:
        failures.append(f"task regressions detected: {', '.join(regressions)}")
    summary = comparison.get("summary_delta", {})
    p95_model = summary.get("p95_model_ms")
    if (
        args.max_p95_model_ms_growth is not None
        and p95_model != "unavailable"
        and float(p95_model) > args.max_p95_model_ms_growth
    ):
        failures.append(
            f"p95 model latency grew by {float(p95_model):.1f}ms > {args.max_p95_model_ms_growth:.1f}ms"
        )
    avg_tokens = summary.get("avg_tokens")
    if (
        args.max_avg_token_growth is not None
        and avg_tokens != "unavailable"
        and float(avg_tokens) > args.max_avg_token_growth
    ):
        failures.append(
            f"average tokens grew by {float(avg_tokens):.1f} > {args.max_avg_token_growth:.1f}"
        )
    avg_tool_calls = summary.get("avg_tool_calls")
    if (
        args.max_avg_tool_calls_growth is not None
        and avg_tool_calls != "unavailable"
        and float(avg_tool_calls) > args.max_avg_tool_calls_growth
    ):
        failures.append(
            f"average tool calls grew by {float(avg_tool_calls):.2f} > {args.max_avg_tool_calls_growth:.2f}"
        )
    grader_rates = comparison.get("candidate_grader_rates", {})
    for grader_name in args.require_grader_pass:
        rate = grader_rates.get(grader_name)
        if rate is None:
            failures.append(f"required grader {grader_name!r} missing from candidate report")
            continue
        if float(rate) < 1.0:
            failures.append(f"required grader {grader_name!r} pass rate is {float(rate):.1%}")
    return failures


def _write_baseline_diff_artifact(output_dir: Path, comparison: dict[str, Any]) -> None:
    path = output_dir / "baseline_diff.json"
    path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class _StaticProvider(ModelProvider):
    def __init__(self, turns: list[list[ProviderEvent]]) -> None:
        self._turns = iter(turns)

    @property
    def model(self) -> str:
        return "offline-static"

    @property
    def thinking(self) -> bool:
        return False

    @property
    def reasoning_effort(self) -> str | None:
        return None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **_kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        try:
            events = next(self._turns)
        except StopIteration:
            events = [FinalMessage(content="", stop_reason="end_turn")]
        for event in events:
            yield event


if __name__ == "__main__":
    raise SystemExit(main())
