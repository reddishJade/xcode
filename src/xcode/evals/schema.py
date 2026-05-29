from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EvalMode = Literal["act", "plan", "review"]


@dataclass(frozen=True)
class EvalTask:
    """单个 eval 用例的声明式规格。"""

    id: str
    prompt: str
    mode: EvalMode = "act"
    expected_answer_contains: tuple[str, ...] = ()
    expected_tool_calls: tuple[str, ...] = ()
    disallowed_tool_calls: tuple[str, ...] = ()
    max_tool_errors: int = 0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraderResult:
    name: str
    passed: bool
    details: str = ""


@dataclass(frozen=True)
class TrialResult:
    task_id: str
    trial_id: str
    success: bool
    answer: str
    trace_path: Path
    graders: tuple[GraderResult, ...]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    success: bool
    output_dir: Path
    trials: tuple[TrialResult, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
