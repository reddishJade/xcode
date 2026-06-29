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
from xcode.evals.cli import _compare_report_to_baseline
from xcode.evals.cli import _evaluate_baseline_gates
from xcode.evals.cli import _trial_project_root
from xcode.evals.cli import _task_from_dict
from xcode.evals import EvalRunner, EvalTask

from xcode.evals.runner import _build_run_metrics
from xcode.evals.sandbox import UnsafeEvalTaskError
from xcode.evals.tasks import SUITES, SUITE_SPECS
from xcode.tests.fixtures import FakeProvider
from xcode.evals.schema import EvalReport, EvalTaskSchemaError, TrialResult
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
            manifest = json.loads((Path(tmp) / "run_manifest.json").read_text(encoding="utf-8"))
            assert manifest["task_ids"] == ["echo-task"]
            assert manifest["trace_schema_version"] == 1
            assert manifest["wall_clock_ms"] != "unavailable"
            assert manifest["termination_reasons"] != "unavailable"
            report_data = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
            metrics = report_data["trials"][0]["metrics"]
            assert metrics["input_tokens"] == "unavailable"
            assert metrics["output_tokens"] == "unavailable"
            history_root = Path(tmp).parent
            assert (history_root / "run_index.jsonl").exists()
            assert (history_root / "trend_summary.json").exists()

    def test_eval_runner_keeps_tool_policy_graders_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(
                id="no-echo",
                prompt="do not run echo",
                disallowed_tool_calls=("echo",),
                expected_answer_contains=("finished",),
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=_tool_app,
                output_dir=Path(tmp),
            )

            report = runner.run()

            assert report.success
            assert report.trials[0].success
            failing = [
                grader.name for grader in report.trials[0].graders if not grader.passed
            ]
            assert "disallowed_tool:echo" in failing
            disallowed = next(
                grader
                for grader in report.trials[0].graders
                if grader.name == "disallowed_tool:echo"
            )
            assert not disallowed.required

    def test_eval_runner_records_tool_policy_state_and_trajectory_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(
                id="policy-metrics",
                prompt="find EvalRunner and read the file",
                expected_tool_calls=("grep_search", "read_file", "read_file"),
                expected_answer_contains=("EvalRunner",),
                metadata={
                    "tool_policy": {
                        "ordered_tools": ("grep_search", "read_file"),
                        "argument_contains": (
                            {"tool": "grep_search", "arguments": {"query": "EvalRunner"}},
                            {
                                "tool": "read_file",
                                "arguments": {"path": "src/xcode/evals/runner.py"},
                            },
                        ),
                        "result_contains": (
                            {"tool": "grep_search", "substrings": ("EvalRunner",)},
                            {"tool": "read_file", "substrings": ("class EvalRunner",)},
                        ),
                        "answer_contains_from_tool": (
                            {"tool": "read_file", "substrings": ("EvalRunner",)},
                        ),
                    }
                },
            )

            report = EvalRunner(
                tasks=(task,),
                app_factory=_offline_app_factory,
                output_dir=Path(tmp),
            ).run()

            assert report.success
            trial = report.trials[0]
            grader_names = {grader.name for grader in trial.graders}
            assert "tool_policy:ordered_tools" in grader_names
            assert "tool_policy:arguments:1:grep_search" in grader_names
            assert "tool_policy:result:2:read_file" in grader_names
            assert "tool_policy:adopted_result:1:read_file" in grader_names
            assert trial.metrics["first_expected_tool_step"] == 1
            assert trial.metrics["repeated_tool_call_count"] == 1
            assert trial.metrics["unexpected_tool_call_count"] == 0

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
            persisted = MemoryManager(
                root,
                user_memory_file=root / "home" / ".xcode" / "memory" / "MEMORY.md",
            ).read_memory_records(layer="project")
            assert persisted[0].retrieval_count == 1
            assert persisted[0].injection_count == 1
            assert persisted[0].adoption_count == 1
            assert persisted[0].success_count == 1
            assert persisted[0].utility == 1.0
            assert persisted[0].last_outcome == "success"
            trace_types = [event["type"] for event in trial.metrics["memory_trace"]]
            assert "used" in trace_types

    def test_eval_runner_persists_explicit_memory_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            task = EvalTask(
                id="memory-reference",
                prompt="provider timeout retry",
                expected_answer_contains=("Provider timeout retry",),
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _memory_app(
                    root,
                    answer="Use Provider timeout retry here.",
                ),
                output_dir=Path(tmp) / "run",
            )

            report = runner.run()

            assert report.success
            persisted = MemoryManager(
                root,
                user_memory_file=root / "home" / ".xcode" / "memory" / "MEMORY.md",
            ).read_memory_records(layer="project")
            assert persisted[0].reference_count == 1
            assert persisted[0].adoption_count == 1
            assert persisted[0].success_count == 1

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
                "llm_judge_required": True,
            }
        )

        assert task.llm_judge_criteria == ("criteria one", "criteria two")
        assert task.llm_judge_required is True

    def test_task_from_dict_rejects_unknown_fields(self) -> None:
        with pytest.raises(EvalTaskSchemaError, match=r"tasks\[0\]\.unknown"):
            _task_from_dict(
                {
                    "id": "bad-task",
                    "prompt": "answer",
                    "unknown": True,
                }
            )

    def test_task_from_dict_rejects_invalid_llm_judge_required_type(self) -> None:
        with pytest.raises(
            EvalTaskSchemaError,
            match=r"tasks\[0\]\.llm_judge_required: expected boolean",
        ):
            _task_from_dict(
                {
                    "id": "bad-judge-required",
                    "prompt": "answer",
                    "llm_judge_required": "yes",
                }
            )

    def test_task_from_dict_accepts_tool_policy_and_fault_injection_metadata(self) -> None:
        task = _task_from_dict(
            {
                "id": "fault-task",
                "prompt": "recover",
                "metadata": {
                    "tool_policy": {
                        "ordered_tools": ["grep_search", "read_file"],
                    },
                    "fault_injection": {
                        "scenario": "wrong_path_recovery",
                    },
                },
            }
        )

        assert task.metadata["tool_policy"]["ordered_tools"] == [
            "grep_search",
            "read_file",
        ]
        assert task.metadata["fault_injection"]["scenario"] == "wrong_path_recovery"

    def test_coding_fixture_suite_is_sandboxed_and_validated(self) -> None:
        tasks = SUITES["coding-fixture"]

        assert len(tasks) >= 1
        for task in tasks:
            assert "fixture_dir" in task.metadata
            assert "validation" in task.metadata
            assert task.metadata["validation"]["commands"]

    def test_memory_suite_registers_on_off_pair(self) -> None:
        tasks = SUITES["memory"]

        assert len(tasks) == 4
        configs = [task.metadata["memory_eval"] for task in tasks]
        assert {config["mode"] for config in configs} == {"on", "off"}
        assert {config["comparison_group"] for config in configs} == {
            "provider-timeout-retry",
            "provider-timeout-conflict",
        }
        assert any(config.get("offline_memory_blocks") for config in configs)
        conflict = next(
            config
            for config in configs
            if config["comparison_group"] == "provider-timeout-conflict"
            and config["mode"] == "on"
        )
        assert conflict["stale_or_conflicting_titles"] == ("Old timeout workaround",)
        assert len(conflict["offline_memory_blocks"]) == 2

    def test_fault_injection_suite_registers_recovery_tasks(self) -> None:
        tasks = SUITES["fault-injection"]

        assert len(tasks) == 3
        assert SUITE_SPECS["fault-injection"].kind == "regression"
        scenarios = {task.metadata["fault_injection"]["scenario"] for task in tasks}
        assert scenarios == {
            "command_failure_retry",
            "wrong_path_recovery",
            "provider_abort_degrade",
        }

    def test_all_suite_excludes_real_coding_fixtures(self) -> None:
        all_tasks = SUITES["all"]

        assert len(all_tasks) >= 1
        for task in all_tasks:
            assert "fixture_dir" not in task.metadata
            assert "validation" not in task.metadata
        assert SUITE_SPECS["all"].kind == "regression"
        assert SUITE_SPECS["capability"].kind == "capability"

    def test_eval_cli_lists_builtin_suites(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--list-suites"])

        assert exit_code == 0
        text = output.getvalue()
        assert "coding-fixture" in text
        assert "memory" in text
        assert "tool-policy" in text
        assert "regression" in text
        assert "capability" in text

    def test_eval_cli_shows_suite_tasks(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--show-suite", "coding-fixture"])

        assert exit_code == 0
        text = output.getvalue()
        assert "tiny-calculator-subtract" in text
        assert "validation_commands" in text

    def test_eval_cli_runs_memory_suite_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = eval_main(["--suite", "memory", "--output-dir", tmp])

            assert exit_code == 0
            text = output.getvalue()
            assert "Memory on/off:" in text
            assert "negative_migration" in text

    def test_eval_cli_runs_fault_injection_suite_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = eval_main(["--suite", "fault-injection", "--output-dir", tmp])

        assert exit_code == 0

    def test_eval_cli_lists_external_benchmarks(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--list-benchmarks"])

        assert exit_code == 0
        text = output.getvalue()
        assert "swebench-lite" in text
        assert "terminal-bench" in text
        assert "catalog-only" in text
        assert "integrated" in text

    def test_eval_cli_baseline_diff_marks_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = EvalTask(
                id="baseline-task",
                prompt="answer",
                expected_answer_contains=("done",),
                capability="tool.read",
            )
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: XcodeApp(
                    agent=StructuredAgent(
                        provider=FakeProvider(
                            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")]
                        ),
                        registry=(),
                    )
                ),
                output_dir=output_dir / "baseline",
            )
            baseline = runner.run()
            candidate = EvalReport(
                run_id="candidate",
                success=False,
                output_dir=output_dir / "candidate",
                tasks=(task,),
                trials=(
                    TrialResult(
                        task_id="baseline-task",
                        trial_id="baseline-task-1",
                        success=False,
                        answer="",
                        trace_path=Path("trace.jsonl"),
                        graders=(),
                    ),
                ),
            )

            diff = _compare_report_to_baseline(
                candidate,
                json.loads((baseline.output_dir / "report.json").read_text(encoding="utf-8")),
            )

            assert diff["regression"] == ["baseline-task"]
            assert "avg_tool_calls" in diff["summary_delta"]
            assert "task_details" in diff
            assert diff["task_details"]["baseline-task"]["candidate"]["failure_categories"] == []

    def test_baseline_gate_fails_on_regression(self) -> None:
        args = type(
            "Args",
            (),
            {
                "fail_on_regression": True,
                "max_p95_model_ms_growth": None,
                "max_avg_token_growth": None,
                "max_avg_tool_calls_growth": None,
                "require_grader_pass": [],
            },
        )()

        failures = _evaluate_baseline_gates(
            {
                "regression": ["shared-task"],
                "summary_delta": {},
                "candidate_grader_rates": {},
            },
            args,
        )

        assert failures == ["task regressions detected: shared-task"]

    def test_baseline_diff_artifact_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_dir = Path(tmp) / "baseline"
            candidate_dir = Path(tmp) / "candidate"
            baseline_task = EvalTask(
                id="shared-task",
                prompt="answer",
                expected_answer_contains=("done",),
            )
            baseline_runner = EvalRunner(
                tasks=(baseline_task,),
                app_factory=lambda _task, _trial: XcodeApp(
                    agent=StructuredAgent(
                        provider=FakeProvider(
                            [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")]
                        ),
                        registry=(),
                    )
                ),
                output_dir=baseline_dir,
            )
            baseline_runner.run()

            exit_code = eval_main(
                [
                    "--tasks",
                    str(_write_tasks_json(candidate_dir / "tasks.json", [baseline_task.to_dict()])),
                    "--output-dir",
                    str(candidate_dir),
                    "--baseline",
                    str(baseline_dir),
                ]
            )

            assert exit_code == 0
            diff_path = candidate_dir / "baseline_diff.json"
            assert diff_path.exists()
            diff = json.loads(diff_path.read_text(encoding="utf-8"))
            assert "task_details" in diff
            assert "shared-task" in diff["task_details"]

    def test_offline_factory_replays_all_expected_tool_calls(self) -> None:
        task = EvalTask(
            id="multi-offline",
            prompt="do two things",
            expected_tool_calls=("grep_search", "read_file"),
            expected_answer_contains=("done",),
        )

        report = EvalRunner(
            tasks=(task,),
            app_factory=_offline_app_factory,
        ).run()

        grader_names = {grader.name for grader in report.trials[0].graders}
        assert "expected_tool:grep_search" in grader_names
        assert "expected_tool:read_file" in grader_names

    def test_eval_cli_prints_failed_trial_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(
                id="bad-answer",
                prompt="do not run echo",
                expected_answer_contains=("missing",),
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
            assert "answer_contains:missing" in text
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


def _memory_app(project_root: Path, *, answer: str = "done") -> XcodeApp:
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
        TextDelta(chunk=answer),
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
            memory_manager=manager,
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


def _write_tasks_json(path: Path, tasks: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    pytest.main()
