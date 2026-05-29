from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.events import (
    FinalMessage,
    TextDelta,
    ToolCall,
    ToolCallReady,
)
from xcode.harness.app import XcodeApp
from xcode.harness.config import AgentConfig
from xcode.harness.skills import ToolSpec
from xcode.evals.cli import _trial_project_root
from xcode.evals import EvalRunner, EvalTask
from xcode.tests.fixtures import FakeProvider


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


def _tool_app(_task: EvalTask, _trial_index: int) -> XcodeApp:
    responses = [
        [ToolCallReady([ToolCall("a", "echo", "hello")]), FinalMessage("", "end_turn")],
        [TextDelta("finished"), FinalMessage("", "end_turn")],
    ]
    tool = ToolSpec("echo", "Echo input.", "text", lambda value: value)
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(responses),
            registry=(tool,),
            config=AgentConfig(max_steps=2),
        )
    )


def _text_app(_task: EvalTask, _trial_index: int) -> XcodeApp:
    return XcodeApp(
        agent=StructuredAgent(
            provider=FakeProvider(
                [TextDelta("async ok"), FinalMessage("", "end_turn")]
            ),
            registry=(),
        )
    )


def _editing_app(project_root: Path) -> XcodeApp:
    def edit_file(_value: str) -> str:
        path = project_root / "math_utils.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n\ndef subtract(left, right):\n    return left - right\n",
            encoding="utf-8",
        )
        return "edited"

    tool = ToolSpec("edit_file", "Edit file.", "text", edit_file)
    responses = [
        [
            ToolCallReady([ToolCall("edit", "edit_file", "math_utils.py")]),
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


if __name__ == "__main__":
    unittest.main()
