from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from xcode.harness.app import XcodeApp
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.config import AgentConfig
from xcode.harness.memory import MemoryManager, MemoryTraceEvent
from xcode.harness.skills import ToolSpec
from xcode.evals.benchmarks import load_benchmark
from xcode.evals.cli import _print_failed_trials
from xcode.evals.cli import _offline_app_factory
from xcode.evals.cli import main as eval_main
from xcode.evals.cli import _trial_project_root
from xcode.evals.cli import _task_from_dict
from xcode.evals import EvalRunner, EvalTask

from xcode.evals.runner import _build_run_metrics
from xcode.evals.sandbox import UnsafeEvalTaskError
from xcode.evals.tasks import SUITES
from xcode.tests.fixtures import FakeProvider
from xcode.evals.schema import TrialResult
from xcode.evals.runner import _build_memory_metrics
import pytest

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}
PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


class EvalPipelineTests:
    def test_offline_app_returns_declared_expected_answer(self) -> None:
        task = EvalTask(
            id="offline-answer",
            prompt="return the marker",
            expected_answer_contains=("expected-marker",),
        )

        answer = _offline_app_factory(task, 0).ask(task.prompt)

        assert answer == "expected-marker"

    def test_eval_runner_records_trace_and_passes_graders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(
                id="echo-task",
                prompt="run echo",
                expected_answer_contains=("finished",),
                expected_tool_calls=("echo",),
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=_tool_app,
                output_dir=Path(tmp),
            )

            report = runner.run()

            assert report.success
            assert report.metrics["trial_count"] == 1
            trial = report.trials[0]
            assert trial.trace_path.exists()
            assert (Path(tmp) / "report.json").exists()
            assert (Path(tmp) / "report.html").exists()
            records = [
                json.loads(line)
                for line in trial.trace_path.read_text(encoding="utf-8").splitlines()
            ]
            assert "tool_use" in [record["type"] for record in records]
            assert "final" in [record["type"] for record in records]

    def test_eval_runner_reports_disallowed_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(
                id="no-echo",
                prompt="do not run echo",
                disallowed_tool_calls=("echo",),
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=_tool_app,
                output_dir=Path(tmp),
            )

            report = runner.run()

            assert not (report.success)
            assert not (report.trials[0].success)
            failing = [
                grader.name for grader in report.trials[0].graders if not grader.passed
            ]
            assert "disallowed_tool:echo" in failing

    def test_eval_runner_arun_works_inside_event_loop(self) -> None:
        async def main():
            with tempfile.TemporaryDirectory() as tmp:
                task = EvalTask(
                    id="async-task",
                    prompt="answer",
                    expected_answer_contains=("async ok",),
                )
                runner = EvalRunner(
                    tasks=(task,),
                    app_factory=_text_app,
                    output_dir=Path(tmp),
                )
                return await runner.arun()

        report = asyncio.run(main())

        assert report.success

    def test_eval_runner_sync_run_rejects_active_event_loop(self) -> None:
        async def main():
            runner = EvalRunner(tasks=(), app_factory=_text_app)
            with pytest.raises(RuntimeError) as exc_info:
                runner.run()
            return str(exc_info.value)

        message = asyncio.run(main())

        assert "use arun" in message

    def test_eval_runner_records_file_change_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            (root / "math_utils.py").write_text(
                "def add(left, right):\n    return left + right\n",
                encoding="utf-8",
            )
            task = EvalTask(
                id="code-change",
                prompt="add subtract",
                expected_answer_contains=("done",),
                metadata={
                    "evidence": {
                        "files": [
                            {
                                "path": "math_utils.py",
                                "changed": True,
                                "contains": ["def subtract", "return left - right"],
                            }
                        ]
                    }
                },
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _editing_app(root),
                output_dir=Path(tmp) / "run",
            )

            report = runner.run()

            assert report.success
            trial = report.trials[0]
            assert "file_evidence" in trial.metrics
            assert trial.metrics["project_root"] == str(root)
            grader_names = {grader.name for grader in trial.graders}
            assert "file_changed:math_utils.py" in grader_names
            assert "file_contains:math_utils.py:def subtract" in grader_names

    def test_eval_runner_grades_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            task = EvalTask(
                id="validation-task",
                prompt="run validation",
                expected_answer_contains=("done",),
                metadata={
                    "validation": {
                        "commands": [
                            [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; assert Path('ok.txt').read_text() == 'ok'",
                            ]
                        ],
                        "timeout_seconds": 5,
                    }
                },
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _validation_app(root),
                output_dir=Path(tmp) / "run",
            )

            report = runner.run()

            assert report.success
            trial = report.trials[0]
            assert "validation" in trial.metrics
            grader_names = {grader.name for grader in trial.graders}
            assert "validation_command:1" in grader_names

    def test_eval_runner_records_model_patch_from_git_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            (root / "math_utils.py").write_text(
                "def add(left, right):\n    return left + right\n",
                encoding="utf-8",
            )
            task = EvalTask(
                id="code-change",
                prompt="add subtract",
                expected_answer_contains=("done",),
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _editing_app(root),
                output_dir=Path(tmp) / "run",
            )

            with patch("xcode.evals.runner.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "diff --git a/math_utils.py b/math_utils.py\n"
                report = runner.run()

            trial = report.trials[0]
            assert "model_patch" in trial.metrics
            assert "math_utils.py" in trial.metrics["model_patch"]

    def test_eval_runner_reports_memory_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            task = EvalTask(
                id="memory-metrics",
                prompt="provider timeout retry",
                expected_answer_contains=("done",),
                metadata={
                    "memory_eval": {
                        "expected_titles": ("Provider timeout retry",),
                    }
                },
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _memory_app(root),
                output_dir=Path(tmp) / "run",
            )

            report = runner.run()

            assert report.success
            trial = report.trials[0]
            assert trial.metrics["memory_recall_at_k"] == 1.0
            assert trial.metrics["memory_mrr"] == 1.0
            assert trial.metrics["memory_irrelevant_injection_rate"] == 0.0
            assert trial.metrics["memory_injected_count"] == 1
            assert trial.metrics["memory_injected_tokens"] > 0
            assert len(trial.metrics["memory_trace"]) >= 2
            assert report.metrics["memory_recall_at_k_mean"] == 1.0
            assert report.metrics["memory_mrr_mean"] == 1.0
            assert report.metrics["memory_irrelevant_injection_rate_mean"] == 0.0

    def test_build_memory_metrics_scores_expected_and_stale_titles(self) -> None:
        task = EvalTask(
            id="memory-metric-formula",
            prompt="provider timeout retry",
            metadata={
                "memory_eval": {
                    "expected_titles": ("Provider timeout retry",),
                    "stale_or_conflicting_titles": ("Old timeout workaround",),
                }
            },
        )
        trace = (
            MemoryTraceEvent(
                type="retrieved",
                memory_id="mem_expected",
                layer="project",
                title="Provider timeout retry",
                score=1.0,
                latency_ms=5.0,
            ),
            MemoryTraceEvent(
                type="retrieved",
                memory_id="mem_stale",
                layer="project",
                title="Old timeout workaround",
                score=1.0,
                latency_ms=5.0,
            ),
            MemoryTraceEvent(
                type="injected",
                memory_id="mem_expected",
                layer="project",
                title="Provider timeout retry",
                score=1.0,
                token_count=12,
            ),
            MemoryTraceEvent(
                type="injected",
                memory_id="mem_stale",
                layer="project",
                title="Old timeout workaround",
                score=1.0,
                token_count=10,
            ),
        )

        metrics = _build_memory_metrics(task, trace)

        assert metrics["memory_recall_at_k"] == 1.0
        assert metrics["memory_mrr"] == 1.0
        assert metrics["memory_irrelevant_injection_rate"] == 0.5
        assert metrics["memory_stale_conflict_retrieval_rate"] == 0.5
        assert metrics["memory_injected_tokens"] == 22

    def test_build_run_metrics_reports_memory_on_off_ablation(self) -> None:
        tasks = (
            EvalTask(
                id="memory-on",
                prompt="task",
                metadata={
                    "memory_eval": {
                        "comparison_group": "provider-timeout",
                        "mode": "on",
                    }
                },
            ),
            EvalTask(
                id="memory-off",
                prompt="task",
                metadata={
                    "memory_eval": {
                        "comparison_group": "provider-timeout",
                        "mode": "off",
                    }
                },
            ),
            EvalTask(
                id="memory-on-regress",
                prompt="task",
                metadata={
                    "memory_eval": {
                        "comparison_group": "retry-regress",
                        "mode": "on",
                    }
                },
            ),
            EvalTask(
                id="memory-off-regress",
                prompt="task",
                metadata={
                    "memory_eval": {
                        "comparison_group": "retry-regress",
                        "mode": "off",
                    }
                },
            ),
        )
        trials = [
            TrialResult(
                task_id="memory-on",
                trial_id="memory-on-1",
                success=True,
                answer="",
                trace_path=Path("trace-on-1.jsonl"),
                graders=(),
                metrics={"tool_calls": 1},
            ),
            TrialResult(
                task_id="memory-off",
                trial_id="memory-off-1",
                success=False,
                answer="",
                trace_path=Path("trace-off-1.jsonl"),
                graders=(),
                metrics={"tool_calls": 3},
            ),
            TrialResult(
                task_id="memory-on-regress",
                trial_id="memory-on-regress-1",
                success=False,
                answer="",
                trace_path=Path("trace-on-2.jsonl"),
                graders=(),
                metrics={"tool_calls": 2},
            ),
            TrialResult(
                task_id="memory-off-regress",
                trial_id="memory-off-regress-1",
                success=True,
                answer="",
                trace_path=Path("trace-off-2.jsonl"),
                graders=(),
                metrics={"tool_calls": 1},
            ),
        ]

        metrics = _build_run_metrics(tasks, trials)

        assert metrics["memory_ablation_pair_count"] == 2
        assert metrics["memory_on_success_rate"] == 0.5
        assert metrics["memory_off_success_rate"] == 0.5
        assert metrics["memory_success_delta"] == 0.0
        assert metrics["memory_tool_call_delta_mean"] == -0.5
        assert metrics["memory_negative_migration_rate"] == 0.5

    def test_trial_project_root_copies_fixture_to_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            fixture = base / "examples" / "eval" / "fixtures" / "tiny"
            fixture.mkdir(parents=True)
            (fixture / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            task = EvalTask(
                id="fixture-task",
                prompt="edit fixture",
                metadata={"fixture_dir": "examples/eval/fixtures/tiny"},
            )

            root = _trial_project_root(
                task,
                0,
                base_root=base,
                output_dir=Path(tmp) / "runs",
            )

            assert (root / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
            assert "fixture-task-1" in str(root)

    def test_trial_project_root_rejects_unisolated_real_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(id="unsafe-task", prompt="edit current repo")

            with pytest.raises(UnsafeEvalTaskError):
                _trial_project_root(
                    task,
                    0,
                    base_root=Path(tmp),
                    output_dir=Path(tmp) / "runs",
                )

    def test_trial_project_root_allows_explicit_project_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task = EvalTask(id="unsafe-task", prompt="edit current repo")

            root = _trial_project_root(
                task,
                0,
                base_root=base,
                output_dir=base / "runs",
                allow_project_mutation=True,
            )

            assert root == base.resolve()

    def test_task_from_dict_normalizes_llm_judge_criteria(self) -> None:
        task = _task_from_dict(
            {
                "id": "judge-task",
                "prompt": "answer",
                "llm_judge_criteria": ["criteria one", "criteria two"],
            }
        )

        assert task.llm_judge_criteria == ("criteria one", "criteria two")

    def test_coding_fixture_suite_is_sandboxed_and_validated(self) -> None:
        tasks = SUITES["coding-fixture"]

        assert len(tasks) >= 1
        for task in tasks:
            assert "fixture_dir" in task.metadata
            assert "validation" in task.metadata
            assert task.metadata["validation"]["commands"]

    def test_all_suite_excludes_real_coding_fixtures(self) -> None:
        all_tasks = SUITES["all"]

        assert len(all_tasks) >= 1
        for task in all_tasks:
            assert "fixture_dir" not in task.metadata
            assert "validation" not in task.metadata

    def test_eval_cli_lists_builtin_suites(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--list-suites"])

        assert exit_code == 0
        text = output.getvalue()
        assert "coding-fixture" in text
        assert "tool-policy" in text

    def test_eval_cli_shows_suite_tasks(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--show-suite", "coding-fixture"])

        assert exit_code == 0
        text = output.getvalue()
        assert "tiny-calculator-subtract" in text
        assert "validation_commands" in text

    def test_eval_cli_lists_external_benchmarks(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--list-benchmarks"])

        assert exit_code == 0
        text = output.getvalue()
        assert "swebench-lite" in text
        assert "terminal-bench" in text

    def test_eval_cli_prints_failed_trial_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(
                id="no-echo",
                prompt="do not run echo",
                disallowed_tool_calls=("echo",),
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=_tool_app,
                output_dir=Path(tmp),
            )
            report = runner.run()
            output = io.StringIO()

            with redirect_stdout(output):
                _print_failed_trials(report)

            text = output.getvalue()
            assert "Failures:" in text
            assert "disallowed_tool:echo" in text
            assert "trace:" in text

    def test_load_benchmark_limit_slices_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "humaneval.jsonl"
            lines = []
            for i in range(5):
                lines.append(
                    json.dumps(
                        {
                            "task_id": f"HumanEval/{i}",
                            "prompt": f"def foo{i}(): pass",
                            "entry_point": f"foo{i}",
                            "test": f"assert foo{i}()",
                        }
                    )
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            all_tasks = load_benchmark("humaneval", path)
            limited = load_benchmark("humaneval", path, limit=2)

            assert len(all_tasks) == 5
            assert len(limited) == 2
            assert limited[0].id == "humaneval-HumanEval-0"
            assert limited[1].id == "humaneval-HumanEval-1"

    def test_pass_at_k_uses_unbiased_estimator(self) -> None:
        trials = (
            _trial("task-a", True),
            _trial("task-a", False),
            _trial("task-a", False),
            _trial("task-b", False),
            _trial("task-b", False),
            _trial("task-b", False),
        )

        metrics = _build_run_metrics(
            (EvalTask(id="task-a", prompt="a"), EvalTask(id="task-b", prompt="b")),
            list(trials),
        )

        assert metrics["pass@k"] == "1/2"
        assert metrics["pass@k_rate"] == 0.5


def _tool_app(_task: EvalTask, _trial_index: int) -> XcodeApp:
    responses: list[list[ProviderEvent]] = [
        [
            ToolCallEvent(
                calls=[ToolCall(id="a", name="echo", input={"input": "hello"})]
            ),
            FinalMessage(content="", stop_reason="end_turn"),
        ],
        [TextDelta(chunk="finished"), FinalMessage(content="", stop_reason="end_turn")],
    ]
    tool = ToolSpec(
        "echo",
        "Echo input.",
        "text",
        lambda value: value["input"],
        schema=INPUT_SCHEMA,
    )
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
        )
    )


def _text_app(_task: EvalTask, _trial_index: int) -> XcodeApp:
    responses: list[ProviderEvent] = [
        TextDelta(chunk="async ok"),
        FinalMessage(content="", stop_reason="end_turn"),
    ]
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(),
        )
    )


def _editing_app(project_root: Path) -> XcodeApp:
    def edit_file(_value: dict) -> str:
        path = project_root / "math_utils.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n\ndef subtract(left, right):\n    return left - right\n",
            encoding="utf-8",
        )
        return "edited"

    tool = ToolSpec("edit_file", "Edit file.", "text", edit_file, schema=PATH_SCHEMA)
    responses: list[list[ProviderEvent]] = [
        [
            ToolCallEvent(
                calls=[
                    ToolCall(
                        id="edit", name="edit_file", input={"path": "math_utils.py"}
                    )
                ]
            ),
            FinalMessage(content="", stop_reason="end_turn"),
        ],
        [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")],
    ]
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
            runtime=AgentRuntimeConfig(project_root=project_root),
        ),
        registry=(tool,),
    )


def _validation_app(project_root: Path) -> XcodeApp:
    def write_ok(_value: dict) -> str:
        (project_root / "ok.txt").write_text("ok", encoding="utf-8")
        return "created"

    tool = ToolSpec(
        "write_file",
        "Write validation file.",
        "text",
        write_ok,
        schema=PATH_SCHEMA,
    )
    responses: list[list[ProviderEvent]] = [
        [
            ToolCallEvent(
                calls=[
                    ToolCall(id="write", name="write_file", input={"path": "ok.txt"})
                ]
            ),
            FinalMessage(content="", stop_reason="end_turn"),
        ],
        [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")],
    ]
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
            runtime=AgentRuntimeConfig(project_root=project_root),
        ),
        registry=(tool,),
    )


def _memory_app(project_root: Path) -> XcodeApp:
    manager = MemoryManager(
        project_root,
        user_memory_file=project_root / "home" / ".xcode" / "memory" / "MEMORY.md",
    )
    manager.memory_file.write_text(
        (
            "## Provider timeout retry\n"
            "- Context/Query: Provider timeout retry\n"
            "- Solution: Retry transient provider failures with backoff\n"
            "- Files: src/provider.py\n"
            "- Takeaways: Bound retries and preserve the root cause\n"
        ),
        encoding="utf-8",
    )
    responses: list[ProviderEvent] = [
        TextDelta(chunk="done"),
        FinalMessage(content="", stop_reason="end_turn"),
    ]
    agent = StructuredAgent(
        provider=FakeProvider(responses),
        registry=(),
        runtime=AgentRuntimeConfig(
            project_root=project_root,
            runtime_context_provider=build_runtime_context_provider(
                project_root,
                (),
                memory_manager=manager,
            ),
        ),
    )
    return XcodeApp(agent=agent, memory_manager=manager)


def _trial(task_id: str, success: bool) -> TrialResult:
    return TrialResult(
        task_id=task_id,
        trial_id=f"{task_id}-1",
        success=success,
        answer="",
        trace_path=Path("trace.jsonl"),
        graders=(),
    )


if __name__ == "__main__":
    pytest.main()
