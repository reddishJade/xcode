from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from xcode.harness.agent_runtime import StructuredAgent
from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from xcode.harness.app import XcodeApp
from xcode.harness.config import AgentConfig
from xcode.harness.skills import ToolSpec
from xcode.evals.benchmarks import load_benchmark
from xcode.evals.cli import _trial_project_root
from xcode.evals.cli import _task_from_dict
from xcode.evals import EvalRunner, EvalTask
from xcode.evals.runner import _build_run_metrics
from xcode.evals.sandbox import UnsafeEvalTaskError
from xcode.tests.fixtures import FakeProvider
from xcode.evals.schema import TrialResult


class EvalPipelineTests(unittest.TestCase):
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

            self.assertTrue(report.success)
            self.assertEqual(report.metrics["trial_count"], 1)
            trial = report.trials[0]
            self.assertTrue(trial.trace_path.exists())
            self.assertTrue((Path(tmp) / "report.json").exists())
            self.assertTrue((Path(tmp) / "report.html").exists())
            records = [
                json.loads(line)
                for line in trial.trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("tool_use", [record["type"] for record in records])
            self.assertIn("final", [record["type"] for record in records])

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

            self.assertFalse(report.success)
            self.assertFalse(report.trials[0].success)
            failing = [
                grader.name for grader in report.trials[0].graders if not grader.passed
            ]
            self.assertIn("disallowed_tool:echo", failing)

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

        self.assertTrue(report.success)

    def test_eval_runner_sync_run_rejects_active_event_loop(self) -> None:
        async def main():
            runner = EvalRunner(tasks=(), app_factory=_text_app)
            with self.assertRaises(RuntimeError) as ctx:
                runner.run()
            return str(ctx.exception)

        message = asyncio.run(main())

        self.assertIn("use arun", message)

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

            self.assertTrue(report.success)
            trial = report.trials[0]
            self.assertIn("file_evidence", trial.metrics)
            grader_names = {grader.name for grader in trial.graders}
            self.assertIn("file_changed:math_utils.py", grader_names)
            self.assertIn(
                "file_contains:math_utils.py:def subtract",
                grader_names,
            )

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

            self.assertEqual(
                (root / "app.py").read_text(encoding="utf-8"), "VALUE = 1\n"
            )
            self.assertIn("fixture-task-1", str(root))

    def test_trial_project_root_rejects_unisolated_real_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = EvalTask(id="unsafe-task", prompt="edit current repo")

            with self.assertRaises(UnsafeEvalTaskError):
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

            self.assertEqual(root, base.resolve())

    def test_task_from_dict_normalizes_llm_judge_criteria(self) -> None:
        task = _task_from_dict(
            {
                "id": "judge-task",
                "prompt": "answer",
                "llm_judge_criteria": ["criteria one", "criteria two"],
            }
        )

        self.assertEqual(task.llm_judge_criteria, ("criteria one", "criteria two"))

    def test_load_humaneval_benchmark_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "humaneval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "def add(a, b):",
                        "entry_point": "add",
                        "test": "assert add(1, 2) == 3",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tasks = load_benchmark("humaneval", path)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].id, "humaneval-HumanEval-0")
            self.assertIn("benchmark", tasks[0].tags)
            self.assertIn("add", tasks[0].expected_answer_contains)
            self.assertTrue(tasks[0].llm_judge_criteria)

    def test_load_swebench_lite_benchmark_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swebench.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "instance_id": "repo__issue-1",
                            "repo": "owner/repo",
                            "base_commit": "abc123",
                            "problem_statement": "Fix the failing parser.",
                            "test_patch": "assert parser()",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            tasks = load_benchmark("swebench-lite", path)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].id, "swebench-lite-repo__issue-1")
            self.assertIn("Fix the failing parser.", tasks[0].prompt)
            self.assertEqual(tasks[0].metadata["benchmark"]["repo"], "owner/repo")

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

        self.assertEqual(metrics["pass@k"], "1/2")
        self.assertEqual(metrics["pass@k_rate"], 0.5)


def _tool_app(_task: EvalTask, _trial_index: int) -> XcodeApp:
    responses: list[list[ProviderEvent]] = [
        [
            ToolCallEvent([ToolCall("a", "echo", {"input": "hello"})]),
            FinalMessage("", "end_turn"),
        ],
        [TextDelta("finished"), FinalMessage("", "end_turn")],
    ]
    tool = ToolSpec("echo", "Echo input.", "text", lambda value: value["input"])
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
        )
    )


def _text_app(_task: EvalTask, _trial_index: int) -> XcodeApp:
    responses: list[ProviderEvent] = [
        TextDelta("async ok"),
        FinalMessage("", "end_turn"),
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

    tool = ToolSpec("edit_file", "Edit file.", "text", edit_file)
    responses: list[list[ProviderEvent]] = [
        [
            ToolCallEvent([ToolCall("edit", "edit_file", {"path": "math_utils.py"})]),
            FinalMessage("", "end_turn"),
        ],
        [TextDelta("done"), FinalMessage("", "end_turn")],
    ]
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
            project_root=project_root,
        ),
        registry=(tool,),
    )


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
    unittest.main()
