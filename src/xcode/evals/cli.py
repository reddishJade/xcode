from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Callable
import json
from pathlib import Path
import shutil
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
from xcode.ai.types import ToolDefinition
from xcode.ai.providers.protocol import ModelProvider
from xcode.harness.app import XcodeApp, build_app as build_real_app
from xcode.harness.config import AgentConfig
from xcode.harness.config import discover_runtime_config
from xcode.harness.skills import ToolSpec

from .runner import EvalRunner
from .schema import EvalTask


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tasks = _load_tasks(args.tasks) if args.tasks else _offline_smoke_tasks()
    output_dir = args.output_dir or Path.cwd() / ".local" / "eval_runs"
    runner = EvalRunner(
        tasks=tasks,
        app_factory=_real_app_factory(args.project_root, output_dir)
        if args.real
        else _offline_app_factory,
        output_dir=output_dir,
        trials_per_task=args.trials,
    )
    report = runner.run()
    print(f"Eval run: {report.run_id}")
    print(f"Status: {'PASS' if report.success else 'FAIL'}")
    print(f"Report JSON: {report.output_dir / 'report.json'}")
    print(f"Report HTML: {report.output_dir / 'report.html'}")
    return 0 if report.success else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Xcode eval tasks.")
    parser.add_argument(
        "--tasks",
        type=Path,
        help="JSON or JSONL EvalTask file. If omitted, runs offline smoke evals.",
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
    parser.add_argument("--trials", type=int, default=1, help="Trials per task.")
    return parser.parse_args(argv)


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
        )
        return build_real_app(
            project_root=effective_root,
            runtime_config=runtime_config,
            env_files=env_files,
        )

    return build


def _trial_project_root(
    task: EvalTask,
    trial_index: int,
    base_root: Path,
    output_dir: Path,
) -> Path:
    fixture_dir = task.metadata.get("fixture_dir")
    if not fixture_dir:
        return base_root
    fixture_path = Path(str(fixture_dir))
    if not fixture_path.is_absolute():
        fixture_path = base_root / fixture_path
    if not fixture_path.is_dir():
        raise ValueError(f"fixture_dir is not a directory: {fixture_path}")
    sandbox = output_dir / "sandboxes" / f"{task.id}-{trial_index + 1}"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(fixture_path, sandbox)
    return sandbox.resolve()


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
        _messages: list[Message],
        _tools: list[ToolDefinition],
    ) -> AsyncIterator[ProviderEvent]:
        try:
            events = next(self._turns)
        except StopIteration:
            events = [FinalMessage("", "end_turn")]
        for event in events:
            yield event


if __name__ == "__main__":
    raise SystemExit(main())
