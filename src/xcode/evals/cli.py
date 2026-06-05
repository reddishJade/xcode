from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Callable
import json
from pathlib import Path
from typing import Any

from xcode.harness.agent_runtime import StructuredAgent
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
from xcode.harness.config import AgentConfig
from xcode.harness.config import discover_runtime_config
from xcode.harness.skills import ToolSpec
from xcode.harness.observability import HITLResult

from .benchmarks import load_benchmark
from .runner import EvalRunner
from .sandbox import trial_project_root
from .schema import EvalTask
from .tasks import SUITE_DESCRIPTIONS, SUITES


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_suites:
        _print_suite_list()
        return 0
    if args.show_suite:
        return _print_suite_detail(args.show_suite)
    tasks: tuple[EvalTask, ...]
    if args.suite:
        t = SUITES.get(args.suite)
        if t is None:
            available = ", ".join(sorted(SUITES))
            print(f"Unknown suite: {args.suite}. Available: {available}")
            return 1
        tasks = t
    elif args.benchmark:
        if args.benchmark_path is None:
            print("--benchmark-path is required when --benchmark is set")
            return 1
        try:
            tasks = load_benchmark(args.benchmark, args.benchmark_path)
        except ValueError as exc:
            print(str(exc))
            return 1
    elif args.tasks:
        tasks = _load_tasks(args.tasks)
    else:
        tasks = SUITES.get("smoke", ())
    output_dir = args.output_dir or Path.cwd() / ".local" / "eval_runs"
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
    )
    report = runner.run()
    print(f"Eval run: {report.run_id}")
    print(f"Status: {'PASS' if report.success else 'FAIL'}")
    _print_enhanced_summary(report)
    print(f"Report JSON: {report.output_dir / 'report.json'}")
    print(f"Report HTML: {report.output_dir / 'report.html'}")
    print(f"Report CSV:  {report.output_dir / 'report.csv'}")
    return 0 if report.success else 1


_NUMERIC_FIELDS = ("llm_calls", "estimated_prompt_tokens", "model_total_ms", "tool_calls", "tool_errors", "steps")


def _fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _print_enhanced_summary(report) -> None:
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
        all_graders = [g for t in report.trials for g in t.graders]
        passed_g = sum(1 for g in all_graders if g.passed)
        print(f"Graders: {passed_g}/{len(all_graders)} ({grader_rate * 100:.1f}%)")
        _print_grader_bars(m.get("per_grader_pass_rate", {}))
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
    _print_distribution(m)


def _print_task_table(report) -> None:
    trials = report.trials
    if not trials:
        return
    from collections import OrderedDict
    task_map: dict[str, list] = OrderedDict()
    for t in trials:
        task_map.setdefault(t.task_id, []).append(t)
    print(f"\nTasks ({len(task_map)} total):")
    id_w = min(max(max(len(t.task_id) for t in trials) + 2, 12), 30)
    for task_id, task_trials in task_map.items():
        best = next((t for t in task_trials if t.success), task_trials[0])
        g_pass = sum(1 for g in best.graders if g.passed)
        g_total = len(best.graders)
        status = "PASS" if best.success else "FAIL"
        calls = best.metrics.get("tool_calls", "")
        tokens = best.metrics.get("estimated_prompt_tokens", "")
        if isinstance(tokens, int):
            tokens = f"{tokens:,}"
        lat = best.metrics.get("model_total_ms", 0)
        lat_str = _fmt_ms(lat) if lat else ""
        mark = "+" if best.success else "x"
        print(
            f"  {mark} {task_id:<{id_w}} {status:>4}  "
            f"graders {g_pass}/{g_total}  "
            f"calls {str(calls):>3}  "
            f"tokens {str(tokens):>7}  "
            f"{lat_str}"
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
        for r in rows:
            print(r)


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
        "--tasks",
        type=Path,
        help="JSON or JSONL EvalTask file. If omitted, runs smoke suite.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("humaneval", "swebench-lite"),
        help="Load tasks from a local benchmark JSON or JSONL file.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        help="Local JSON or JSONL file for --benchmark.",
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
        "--allow-project-mutation",
        action="store_true",
        help=(
            "Allow --real eval tasks without fixture_dir to modify --project-root. "
            "By default real eval writes must run in sandboxes."
        ),
    )
    parser.add_argument("--trials", type=int, default=1, help="Trials per task.")
    return parser.parse_args(argv)


def _print_suite_list() -> None:
    print("Built-in eval suites:")
    for name in sorted(SUITES):
        tasks = SUITES[name]
        description = SUITE_DESCRIPTIONS.get(name, "")
        has_fixture = any("fixture_dir" in task.metadata for task in tasks)
        real_hint = "real/sandbox" if has_fixture else "offline"
        print(f"  {name:<16} {len(tasks):>2} tasks  {real_hint:<12} {description}")


def _print_suite_detail(name: str) -> int:
    tasks = SUITES.get(name)
    if tasks is None:
        available = ", ".join(sorted(SUITES))
        print(f"Unknown suite: {name}. Available: {available}")
        return 1
    description = SUITE_DESCRIPTIONS.get(name, "")
    print(f"Suite: {name}")
    if description:
        print(f"Description: {description}")
    print(f"Tasks ({len(tasks)}):")
    for task in tasks:
        fixture = task.metadata.get("fixture_dir")
        validation = task.metadata.get("validation", {})
        commands = validation.get("commands", ()) if isinstance(validation, dict) else ()
        print(f"  - {task.id}")
        if fixture:
            print(f"    fixture: {fixture}")
        if commands:
            print(f"    validation_commands: {len(commands)}")
    return 0


def _load_tasks(path: Path) -> tuple[EvalTask, ...]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        raw_items = raw if isinstance(raw, list) else [raw]
    return tuple(_task_from_dict(item) for item in raw_items)


def _task_from_dict(item: dict[str, Any]) -> EvalTask:
    tuple_keys = (
        "expected_answer_contains",
        "expected_tool_calls",
        "disallowed_tool_calls",
        "tags",
        "llm_judge_criteria",
    )
    normalized = dict(item)
    for key in tuple_keys:
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    return EvalTask(**normalized)


def _offline_smoke_tasks() -> tuple[EvalTask, ...]:
    return (
        EvalTask(
            id="offline-answer",
            prompt="Return the offline smoke answer.",
            expected_answer_contains=("offline ok",),
            tags=("offline", "smoke"),
        ),
        EvalTask(
            id="offline-tool",
            prompt="Call echo once, then finish.",
            expected_answer_contains=("finished",),
            expected_tool_calls=("echo",),
            tags=("offline", "tool-use"),
        ),
    )


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
        # eval 模式下自动批准所有工具调用，避免 approval_required 阻塞
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
    provider: _StaticProvider
    tools: tuple[ToolSpec, ...]
    if task.expected_tool_calls:
        tool_name = task.expected_tool_calls[0]
        provider = _StaticProvider(
            [
                [
                    ToolCallEvent(
                        [
                            ToolCall(
                                id=f"{task.id}-call-1",
                                name=tool_name,
                                input={"input": task.prompt},
                            )
                        ]
                    ),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("finished"), FinalMessage("", "end_turn")],
            ]
        )
        tools = (
            ToolSpec(
                name=tool_name,
                description="Offline eval echo tool.",
                input_hint="text",
                handler=lambda value: str(value.get("input", "")),
                read_only=True,
                concurrency_safe=True,
            ),
        )
    else:
        provider = _StaticProvider(
            [[TextDelta("offline ok"), FinalMessage("offline ok", "end_turn")]]
        )
        tools = ()
    return XcodeApp(
        agent=StructuredAgent(
            provider=provider,
            registry=tools,
            config=AgentConfig(max_steps=3),
        ),
        registry=tools,
    )


class _StaticProvider(ModelProvider):
    def __init__(self, turns: list[list[ProviderEvent]]) -> None:
        self._turns = iter(turns)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderEvent]:
        try:
            events = next(self._turns)
        except StopIteration:
            events = [FinalMessage("", "end_turn")]
        for event in events:
            yield event


if __name__ == "__main__":
    raise SystemExit(main())
