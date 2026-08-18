"""类型化工具呈现投影测试。"""

from xcode.agent.types import (
    DiffRenderIntent,
    LocationRenderIntent,
    SubagentRenderIntent,
    TerminalRenderIntent,
)
from xcode.cli.tool_rendering import render_intent_summary


def test_terminal_summary_uses_typed_context_and_output() -> None:
    summary = render_intent_summary(
        TerminalRenderIntent(command="pytest -q", cwd="/project"),
        "12 passed",
    )

    assert summary == "Terminal /project: 12 passed"


def test_diff_summary_uses_files_and_first_line() -> None:
    summary = render_intent_summary(
        DiffRenderIntent(
            patch="diff",
            files=("src/app.py",),
            first_changed_line=8,
        )
    )

    assert summary == "Changed src/app.py:8"


def test_location_summary_uses_line_range() -> None:
    summary = render_intent_summary(
        LocationRenderIntent(path="src/app.py", line_start=10, line_end=20)
    )

    assert summary == "src/app.py:10-20"


def test_subagent_summary_uses_batch_identity() -> None:
    summary = render_intent_summary(
        SubagentRenderIntent(
            batch_id="batch-1",
            run_ids=("run-1", "run-2"),
        )
    )

    assert summary == "Subagents batch-1: 2 run(s)"
