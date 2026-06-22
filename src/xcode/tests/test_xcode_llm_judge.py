"""LLM-as-judge 流式 provider 集成测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from xcode.ai.events import FinalMessage, TextDelta
from xcode.evals import EvalRunner, EvalTask
from xcode.evals.graders import run_llm_judge
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.app import XcodeApp
from xcode.tests.fixtures import FakeProvider
import pytest
class XcodeLlmJudgeTests:
    """验证 judge 使用 StreamProvider 并显式报告 skipped。"""

    def test_run_llm_judge_consumes_stream_provider(self) -> None:
        """judge 直接消费标准 provider 事件流并解析每条标准。"""
        provider = FakeProvider(
            [
                TextDelta(chunk="PASS: 1: complete\nFAIL: 2: missing edge case"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
        )
        task = EvalTask(
            id="judge",
            prompt="solve",
            llm_judge_criteria=("complete", "handles edge cases"),
        )

        graders = asyncio.run(run_llm_judge(task, "answer", [], provider))

        assert [grader.passed for grader in graders] == [True, False]
        assert not (any(grader.skipped for grader in graders))
        assert provider.last_tools == []
        assert "handles edge cases" in str(provider.last_messages)

    def test_eval_runner_applies_streaming_judge_failure(self) -> None:
        """runner 将 judge FAIL 纳入 trial success。"""
        provider = FakeProvider(
            [
                [
                    TextDelta(chunk="candidate answer"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="FAIL: 1: required evidence is missing"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        task = EvalTask(
            id="judge-failure",
            prompt="answer",
            expected_answer_contains=("candidate",),
            llm_judge_criteria=("includes required evidence",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _app(provider),
                output_dir=Path(temp_dir),
            )

            report = runner.run()

        assert not (report.success)
        judge_result = next(
            grader
            for grader in report.trials[0].graders
            if grader.name == "llm_judge:1"
        )
        assert not (judge_result.passed)
        assert not (judge_result.skipped)

    def test_eval_runner_records_unparseable_judge_as_skipped(self) -> None:
        """judge 未返回可解析结果时 report 显式记录 skipped。"""
        provider = FakeProvider(
            [
                TextDelta(chunk="candidate answer"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]
        )
        task = EvalTask(
            id="judge-skipped",
            prompt="answer",
            expected_answer_contains=("candidate",),
            llm_judge_criteria=("includes required evidence",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            runner = EvalRunner(
                tasks=(task,),
                app_factory=lambda _task, _trial: _app(provider),
                output_dir=output_dir,
            )

            report = runner.run()
            report_data = json.loads(
                (output_dir / "report.json").read_text(encoding="utf-8")
            )
            report_html = (output_dir / "report.html").read_text(encoding="utf-8")
            report_csv = (output_dir / "report.csv").read_text(encoding="utf-8")

        assert report.success
        skipped = next(grader for grader in report.trials[0].graders if grader.skipped)
        assert skipped.name == "llm_judge:skipped"
        assert report.metrics["grader_skipped_count"] == 1
        serialized = report_data["trials"][0]["graders"]
        assert any(grader["skipped"] for grader in serialized)
        assert "SKIP" in report_html
        assert "graders_skipped" in report_csv

    def test_missing_judge_provider_returns_explicit_skip(self) -> None:
        """未注入 provider 时返回不影响成败的 skipped grader。"""
        task = EvalTask(
            id="judge-missing",
            prompt="answer",
            llm_judge_criteria=("criterion",),
        )

        graders = asyncio.run(run_llm_judge(task, "answer", [], None))

        assert len(graders) == 1
        assert graders[0].passed
        assert graders[0].skipped
        assert "unavailable" in graders[0].details

def _app(provider: FakeProvider) -> XcodeApp:
    """构建仅使用给定 provider 的最小 eval 应用。"""
    return XcodeApp(agent=StructuredAgent(provider=provider, registry=()))

if __name__ == "__main__":
    pytest.main()
