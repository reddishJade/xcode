"""真实 executor 的装配与 trace 契约测试。

本文件用装配替身验证基础设施，不构成能力 Eval；真实模型证据必须来自正式 Trial。
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from xcode.evals.executor import ExecutorError, RealXcodeExecutor
from xcode.evals.schema import (
    ModelConfig,
    ResourceBudget,
    Task,
    TaskSource,
    Trial,
    Variant,
)
from xcode.harness.agent_runtime.events import FinalStructuredEvent
from xcode.harness.agent_runtime.result import CodingAgentHarnessResult
from xcode.harness.config import XcodeRuntimeConfig


def _task() -> Task:
    return Task(
        task_id="task-1",
        dataset_version="v1",
        prompt="修复真实问题。",
        source=TaskSource(
            kind="git_history",
            repository="xcode",
            revision="parent",
            license="MIT",
        ),
        verifier_id="hidden-1",
        allowed_paths=("src",),
        budget=ResourceBudget(wall_time_seconds=30, model_calls=5, tool_calls=20),
    )


def _trial() -> Trial:
    return Trial(
        trial_id="trial-1",
        experiment_id="experiment-1",
        task_id="task-1",
        dataset_version="v1",
        variant=Variant(
            variant_id="full",
            harness_revision="current",
            capabilities={"full": True},
        ),
        model=ModelConfig(provider="real", model="real-model"),
        budget=_task().budget,
        repetition=0,
        workspace_revision="parent",
        command=("xcode-eval", "run"),
    )


class _App:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(cancel_active_run=lambda _reason: None)
        self.closed = False

    async def aask_stream(self, _prompt: str, mode: str | None = None):
        assert mode == "build"
        result = CodingAgentHarnessResult(
            answer="done",
            messages=[],
            steps=1,
            tool_calls=[],
            metrics={"llm_calls": 1, "tool_calls": 0},
        )
        yield FinalStructuredEvent("final", 1, result)

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_executor_uses_build_app_and_writes_final_trace(tmp_path: Path) -> None:
    app = _App()
    trace = tmp_path / "trace.jsonl"
    with patch("xcode.evals.executor.build_app", return_value=app) as builder:
        execution = await RealXcodeExecutor().run(
            task=_task(),
            trial=_trial(),
            workspace=tmp_path,
            runtime_config=XcodeRuntimeConfig(),
            trace_path=trace,
        )

    builder.assert_called_once()
    assert builder.call_args.kwargs["runtime_config"].agent.max_steps == 5
    assert app.agent.approval_callback(object(), {}).decision == "allow"
    assert execution.result.answer == "done"
    assert execution.usage.model_calls == 1
    assert len(execution.trace_lines) == 1
    assert '"type": "final"' in trace.read_text(encoding="utf-8")
    assert app.closed is True


@pytest.mark.asyncio
async def test_executor_rejects_mismatched_task_and_trial(tmp_path: Path) -> None:
    trial = _trial().model_copy(update={"task_id": "different"})

    with pytest.raises(ExecutorError, match="identity"):
        await RealXcodeExecutor().run(
            task=_task(),
            trial=trial,
            workspace=tmp_path,
            runtime_config=XcodeRuntimeConfig(),
            trace_path=tmp_path / "trace.jsonl",
        )
